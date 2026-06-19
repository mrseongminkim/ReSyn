import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch.nn.init import xavier_uniform_

from config import Set2RegexConfig, StringConfig

from .dataset import PartitionerVocabulary, RegexVocabulary, SegmenterVocabulary, StringVocabulary


class MAB(nn.Module):
    def __init__(self, hidden_size, n_heads, ln=False):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.rff = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        if ln:
            self.ln0 = nn.LayerNorm(hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)

    def forward(self, query, key, key_padding_mask=None):
        output, _ = self.multihead_attention(query, key, key, key_padding_mask=key_padding_mask)
        output = query + output
        output = output if getattr(self, 'ln0', None) is None else self.ln0(output)
        output = output + self.rff(output)
        output = output if getattr(self, 'ln1', None) is None else self.ln1(output)
        return output


class PMA(nn.Module):
    def __init__(self, hidden_size, n_heads, ln=False):
        super().__init__()
        self.seed_vector = nn.Parameter(torch.Tensor(1, 1, hidden_size))
        nn.init.xavier_uniform_(self.seed_vector)
        self.mab = MAB(hidden_size, n_heads, ln=ln)

    def forward(self, x, key_padding_mask=None):
        seed_vector = self.seed_vector.repeat(x.size(0), 1, 1)
        return self.mab(seed_vector, x, key_padding_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, step=None):
        if step is not None:
            return self.pe[:, step : step + 1]
        else:
            return self.pe[:, : x.size(1)]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None, is_inference=False):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        current_kv = (k, v)
        if is_inference:
            att = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y), current_kv


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, memory, memory_key_padding_mask=None, past_kv=None, is_inference=False):
        residual = x
        x = self.ln1(x)
        x, new_kv = self.self_attn(x, past_kv=past_kv, is_inference=is_inference)
        x = residual + x
        residual = x
        x = self.ln2(x)
        attn_out, _ = self.cross_attn(query=x, key=memory, value=memory, key_padding_mask=memory_key_padding_mask)
        x = residual + attn_out
        residual = x
        x = self.ln3(x)
        x = residual + self.mlp(x)
        return x, new_kv


class Segmenter(
    nn.Module,
    PyTorchModelHubMixin,
    library_name='resyn',
    tags=['regex-synthesis', 'segmenter', 'pytorch'],
    repo_url='https://github.com/mrseongminkim/ReSyn',
    paper_url='https://arxiv.org/pdf/2603.24624',
):
    def __init__(self, d_model=256, num_layers=4, num_heads=4, max_len=1_110):
        super().__init__()
        self.enc_vocab = StringVocabulary.get_vocabulary_size()
        self.dec_vocab = SegmenterVocabulary.get_vocabulary_size()
        self.d_model = d_model
        d_ff = 4 * self.d_model
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=max_len)
        self.enc_emb = nn.Embedding(self.enc_vocab, self.d_model, padding_idx=StringVocabulary.pad_token_index)
        self.dec_emb = nn.Embedding(self.dec_vocab, self.d_model, padding_idx=SegmenterVocabulary.pad_token_index)
        enc_layer = nn.TransformerEncoderLayer(self.d_model, num_heads, d_ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, nn.LayerNorm(self.d_model), enable_nested_tensor=False)
        self.decoder_layers = nn.ModuleList([DecoderBlock(self.d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.dec_final_norm = nn.LayerNorm(self.d_model)
        self.fc_out = nn.Linear(self.d_model, self.dec_vocab)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        All layers in the TransformerEncoder / TransformerDecoder are initialized with the same parameters.
        It is recommended to manually initialize the layers after creating the TransformerEncoder / TransformerDecoder instance.
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, input_ids, decoder_input_ids):
        mask = (input_ids == StringVocabulary.pad_token_index).to(input_ids.device)
        input_embeds = self.enc_emb(input_ids)
        input_embeds = input_embeds * math.sqrt(self.d_model) + self.positional_encoding(input_embeds)
        memory = self.encoder(input_embeds, src_key_padding_mask=mask)
        decoder_embeds = self.dec_emb(decoder_input_ids)
        decoder_embeds = decoder_embeds * math.sqrt(self.d_model) + self.positional_encoding(decoder_embeds)
        for layer in self.decoder_layers:
            decoder_embeds, _ = layer(decoder_embeds, memory, mask, past_kv=None, is_inference=False)
        decoder_embeds = self.dec_final_norm(decoder_embeds)
        logits = self.fc_out(decoder_embeds)
        logits = F.log_softmax(logits, dim=-1)
        return logits


class Partitioner(
    nn.Module,
    PyTorchModelHubMixin,
    library_name='resyn',
    tags=['regex-synthesis', 'partitioner', 'pytorch'],
    repo_url='https://github.com/mrseongminkim/ReSyn',
    paper_url='https://arxiv.org/pdf/2603.24624',
):
    def __init__(self, hidden_size=256, n_layers=2, n_heads=8, max_string_length=StringConfig.max_string_length, device=None):
        super().__init__()
        input_vocab_size = StringVocabulary.get_vocabulary_size()
        self.input_pad_token_index = StringVocabulary.pad_token_index
        output_vocab_size = PartitionerVocabulary.get_vocabulary_size()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_string_length = max_string_length
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.character_embedding = nn.Embedding(input_vocab_size, hidden_size, padding_idx=self.input_pad_token_index)
        self.positional_encoding = PositionalEncoding(hidden_size, max_string_length)
        self.encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, n_heads, dim_feedforward=hidden_size * 4, batch_first=True, norm_first=True
        )
        self.character_level_encoder = nn.TransformerEncoder(
            self.encoder_layer, n_layers, nn.LayerNorm(hidden_size), enable_nested_tensor=False
        )
        self.character_level_attention_pooling = PMA(hidden_size, n_heads, ln=True)
        self.string_level_encoder = nn.TransformerEncoder(
            self.encoder_layer, n_layers, nn.LayerNorm(hidden_size), enable_nested_tensor=False
        )

        self.label_embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.decoder_layer = nn.TransformerDecoderLayer(
            hidden_size, n_heads, dim_feedforward=hidden_size * 4, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, n_layers, nn.LayerNorm(hidden_size))
        self.out = nn.Linear(hidden_size, output_vocab_size)

        self._reset_parameters()

    def encode(self, pos: torch.Tensor):
        batch_size, n_strings, string_max_len = pos.shape
        pos = pos.view(batch_size * n_strings, string_max_len)
        mask = torch.eq(pos, self.input_pad_token_index).to(self.device)
        character_embedding = self.character_embedding(pos)
        character_embedding = character_embedding * math.sqrt(self.hidden_size) + self.positional_encoding(character_embedding)
        character_memory = self.character_level_encoder(character_embedding, src_key_padding_mask=mask)
        string_embedding = self.character_level_attention_pooling(character_memory, key_padding_mask=mask)
        string_embedding = string_embedding.view(batch_size, n_strings, self.hidden_size)
        string_embedding = string_embedding * math.sqrt(self.hidden_size) + self.positional_encoding(string_embedding)
        string_embedding = self.string_level_encoder(string_embedding)  # batch_size, n_strings, self.hidden_size
        return string_embedding

    def forward(self, pos: torch.Tensor, decoder_inputs: torch.Tensor):
        memory = self.encode(pos)
        target_mask = nn.Transformer.generate_square_subsequent_mask(decoder_inputs.size(-1)).to(self.device)
        decoder_embedding = self.label_embedding(decoder_inputs)
        decoder_embedding = decoder_embedding * math.sqrt(self.hidden_size) + self.positional_encoding(decoder_embedding)
        output = self.decoder(decoder_embedding, memory, tgt_is_causal=True, tgt_mask=target_mask)
        output = F.log_softmax(self.out(output), dim=-1)
        return output

    def predict(self, pos: torch.Tensor):
        batch_size, n_strings, string_max_len = pos.shape
        memory = self.encode(pos)
        sos_token_index = PartitionerVocabulary.sos_token_index  # \x00
        labels = torch.full((batch_size, n_strings), sos_token_index, dtype=torch.long).to(self.device)
        target_mask = nn.Transformer.generate_square_subsequent_mask(n_strings - 1).to(self.device)
        log_probs = []
        for t in range(n_strings - 1):
            decoder_inputs = labels[:, : t + 1]
            decoder_embedding = self.label_embedding(decoder_inputs)
            decoder_embedding = decoder_embedding * math.sqrt(self.hidden_size) + self.positional_encoding(decoder_embedding)
            output = self.decoder(decoder_embedding, memory, tgt_is_causal=True, tgt_mask=target_mask[: t + 1, : t + 1])
            output = F.log_softmax(self.out(output[:, -1]), dim=-1)
            log_probs.append(output)
            next_token = output.argmax(dim=-1)
            labels[:, t + 1] = next_token
        log_probs = torch.stack(log_probs, dim=1)
        return labels, log_probs

    def _reset_parameters(self):
        """
        All layers in the TransformerEncoder / TransformerDecoder are initialized with the same parameters.
        It is recommended to manually initialize the layers after creating the TransformerEncoder / TransformerDecoder instance.
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class Router(
    nn.Module,
    PyTorchModelHubMixin,
    library_name='resyn',
    tags=['regex-synthesis', 'router', 'pytorch'],
    repo_url='https://github.com/mrseongminkim/ReSyn',
    paper_url='https://arxiv.org/pdf/2603.24624',
):
    def __init__(self, hidden_size=256, n_layers=2, n_heads=8, max_string_length=StringConfig.max_string_length, device=None):
        super().__init__()
        input_vocab_size = StringVocabulary.get_vocabulary_size()
        self.input_pad_token_index = StringVocabulary.pad_token_index
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_string_length = max_string_length
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.character_embedding = nn.Embedding(input_vocab_size, hidden_size, padding_idx=self.input_pad_token_index)
        self.positional_encoding = PositionalEncoding(hidden_size, max_string_length)
        self.encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, n_heads, dim_feedforward=hidden_size * 4, batch_first=True, norm_first=True
        )
        self.character_level_encoder = nn.TransformerEncoder(
            self.encoder_layer, n_layers, nn.LayerNorm(hidden_size), enable_nested_tensor=False
        )
        self.character_level_attention_pooling = PMA(hidden_size, n_heads, ln=True)
        self.string_level_encoder = nn.TransformerEncoder(
            self.encoder_layer, n_layers, nn.LayerNorm(hidden_size), enable_nested_tensor=False
        )
        self.string_level_attention_pooling = PMA(hidden_size, n_heads, ln=True)

        self.classification_head = nn.Linear(hidden_size, 3)  # Concat, Union, No-Op

        self._reset_parameters()

    def forward(self, pos: torch.Tensor):
        batch_size, n_strings, string_max_len = pos.shape
        pos = pos.view(batch_size * n_strings, string_max_len)
        mask = torch.eq(pos, self.input_pad_token_index).to(self.device)
        character_embedding = self.character_embedding(pos)
        character_embedding = character_embedding * math.sqrt(self.hidden_size) + self.positional_encoding(character_embedding)
        character_memory = self.character_level_encoder(character_embedding, src_key_padding_mask=mask)
        string_embedding = self.character_level_attention_pooling(character_memory, key_padding_mask=mask)
        string_embedding = string_embedding.view(batch_size, n_strings, self.hidden_size)
        string_embedding = self.string_level_encoder(string_embedding)
        set_embedding = self.string_level_attention_pooling(string_embedding)  # batch_size, 1, hidden_size
        set_embedding = set_embedding.squeeze(1)  # batch_size, hidden_size
        output = F.log_softmax(self.classification_head(set_embedding), dim=-1)  # batch_size, 3
        return output

    def _reset_parameters(self):
        """
        All layers in the TransformerEncoder / TransformerDecoder are initialized with the same parameters.
        It is recommended to manually initialize the layers after creating the TransformerEncoder / TransformerDecoder instance.
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class Set2Regex(
    nn.Module,
    PyTorchModelHubMixin,
    library_name='resyn',
    tags=['regex-synthesis', 'set2regex', 'pytorch'],
    repo_url='https://github.com/mrseongminkim/ReSyn',
    paper_url='https://arxiv.org/pdf/2603.24624',
):
    def __init__(self, hidden_size=256, n_layers=2, n_heads=8, max_string_length=Set2RegexConfig.max_regex_length + 1, device=None):
        super().__init__()
        input_vocab_size = StringVocabulary.get_vocabulary_size()
        self.input_pad_token_index = StringVocabulary.pad_token_index
        output_vocab_size = RegexVocabulary.get_vocabulary_size()
        self.output_pad_token_index = RegexVocabulary.pad_token_index
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_string_length = max_string_length
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.character_embedding = nn.Embedding(input_vocab_size, hidden_size, padding_idx=self.input_pad_token_index)
        self.positional_encoding = PositionalEncoding(hidden_size, max_string_length * 2)
        self.encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, n_heads, dim_feedforward=hidden_size * 4, batch_first=True, norm_first=True
        )
        self.character_level_encoder = nn.TransformerEncoder(
            self.encoder_layer, n_layers, nn.LayerNorm(hidden_size), enable_nested_tensor=False
        )
        self.character_level_attention_pooling = PMA(hidden_size, n_heads, ln=True)
        self.string_level_encoder = nn.TransformerEncoder(
            self.encoder_layer, n_layers, nn.LayerNorm(hidden_size), enable_nested_tensor=False
        )
        self.string_level_attention_pooling = PMA(hidden_size, n_heads, ln=True)
        self.type_embedding = nn.Embedding(2, hidden_size)

        self.regex_embedding = nn.Embedding(output_vocab_size, hidden_size, padding_idx=self.output_pad_token_index)
        self.decoder_layer = nn.TransformerDecoderLayer(
            hidden_size, n_heads, dim_feedforward=hidden_size * 4, batch_first=True, norm_first=True
        )
        self.set_decoder = nn.TransformerDecoder(self.decoder_layer, n_layers, nn.LayerNorm(hidden_size))
        self.string_decoder = nn.TransformerDecoder(self.decoder_layer, n_layers, nn.LayerNorm(hidden_size))
        self.out = nn.Linear(hidden_size, output_vocab_size)

        self._reset_parameters()

    def encode(self, strings: torch.Tensor, types: torch.Tensor):
        batch_size, n_strings, string_max_len = strings.shape
        strings = strings.view(batch_size * n_strings, string_max_len)
        mask = torch.eq(strings, self.input_pad_token_index).to(self.device)
        character_embedding = self.character_embedding(strings)
        character_embedding = character_embedding * math.sqrt(self.hidden_size) + self.positional_encoding(character_embedding)
        character_embedding = self.character_level_encoder(character_embedding, src_key_padding_mask=mask)
        string_embedding = self.character_level_attention_pooling(character_embedding, key_padding_mask=mask)
        string_embedding = string_embedding.view(batch_size, n_strings, self.hidden_size)
        type_embedding = self.type_embedding(types)
        string_embedding = string_embedding + type_embedding
        string_embedding = self.string_level_encoder(string_embedding)  # batch_size, n_strings, self.hidden_size
        set_embedding = self.string_level_attention_pooling(string_embedding)  # batch_size, 1, self.hidden_size
        return string_embedding, set_embedding

    def forward(self, strings: torch.Tensor, types: torch.Tensor, decoder_inputs: torch.Tensor):
        string_embedding, set_embedding = self.encode(strings, types)
        target_key_padding_mask = torch.eq(decoder_inputs, self.output_pad_token_index).to(self.device)
        target_mask = nn.Transformer.generate_square_subsequent_mask(decoder_inputs.size(-1)).to(self.device)
        decoder_embedding = self.regex_embedding(decoder_inputs)
        decoder_embedding = decoder_embedding * math.sqrt(self.hidden_size) + self.positional_encoding(decoder_embedding)
        output = self.set_decoder(
            decoder_embedding, set_embedding, tgt_key_padding_mask=target_key_padding_mask, tgt_is_causal=True, tgt_mask=target_mask
        )
        output = self.string_decoder(
            output, string_embedding, tgt_key_padding_mask=target_key_padding_mask, tgt_is_causal=True, tgt_mask=target_mask
        )
        output = F.log_softmax(self.out(output), dim=-1)
        return output

    def predict(
        self,
        strings: torch.Tensor,
        types: torch.Tensor,
        max_regex_length: Optional[int] = None,
        *,
        temperature=1.0,
        k=0,
        p=0.0,
        n_candidates=1,
        beam_size=0,
    ):
        batch_size = strings.size(0)

        if max_regex_length is None:
            max_regex_length = Set2RegexConfig.max_regex_length + 2  # <sos>, <eos>

        # Beam search mode: beam_size가 지정되면 그만큼의 후보 반환
        if batch_size == 1 and beam_size > 0:
            return self._beam_search(strings, types, max_regex_length, beam_size, temperature)

        # Sampling mode
        sampling = False
        string_embedding, set_embedding = self.encode(strings, types)
        if batch_size == 1 and (k != 0 or p != 0.0):
            sampling = True
        if sampling:
            batch_size = n_candidates
            string_embedding = string_embedding.expand(batch_size, -1, -1)  # n_candidates, n_strings, hidden_size
            set_embedding = set_embedding.expand(batch_size, -1, -1)  # n_candidates, 1, hidden_size
        regexes = torch.full((batch_size, max_regex_length), self.output_pad_token_index, dtype=torch.long).to(self.device)  # <sos>, <eos>
        regexes[:, 0] = RegexVocabulary.sos_token_index
        target_mask = nn.Transformer.generate_square_subsequent_mask(max_regex_length - 1).to(self.device)
        log_probs = []
        for t in range(max_regex_length - 1):
            decoder_inputs = regexes[:, : t + 1]
            target_key_padding_mask = torch.eq(decoder_inputs, self.output_pad_token_index).to(self.device)
            decoder_embedding = self.regex_embedding(decoder_inputs)
            decoder_embedding = decoder_embedding * math.sqrt(self.hidden_size) + self.positional_encoding(decoder_embedding)
            output = self.set_decoder(
                decoder_embedding,
                set_embedding,
                tgt_key_padding_mask=target_key_padding_mask,
                tgt_is_causal=True,
                tgt_mask=target_mask[: t + 1, : t + 1],
            )
            output = self.string_decoder(
                output,
                string_embedding,
                tgt_key_padding_mask=target_key_padding_mask,
                tgt_is_causal=True,
                tgt_mask=target_mask[: t + 1, : t + 1],
            )
            final_logits = self.out(output[:, -1])  # batch_size, output_vocab_size
            final_log_probs = F.log_softmax(final_logits, dim=-1)  # batch_size, output_vocab_size
            log_probs.append(final_log_probs)
            if sampling:
                next_token = self._sample(final_log_probs, temperature=temperature, k=k, p=p).squeeze(-1)  # batch_size
            else:
                next_token = final_log_probs.argmax(dim=-1)
            regexes[:, t + 1] = next_token
            if batch_size == 1 and next_token.item() == RegexVocabulary.eos_token_index:
                break
        log_probs = torch.stack(log_probs, dim=1)  # batch_size, max_regex_length - 1, output_vocab_size
        if sampling:
            actual_outputs = regexes[:, 1:]
            chosen_log_probs = torch.gather(log_probs, 2, actual_outputs.unsqueeze(-1)).squeeze(-1)
            is_eos = actual_outputs == RegexVocabulary.eos_token_index
            shifted_cumsum = is_eos.cumsum(dim=1) - is_eos.int()
            valid_token_mask = shifted_cumsum == 0
            summed_scores = chosen_log_probs.masked_fill(~valid_token_mask, 0.0).sum(dim=1)
            lengths = valid_token_mask.sum(dim=1).float().clamp(min=1)
            scores = summed_scores / lengths
            _, sorted_indices = torch.sort(scores, descending=True)
            regexes = regexes[sorted_indices]
            log_probs = log_probs[sorted_indices]
        return regexes[:, 1:].view(batch_size, -1), log_probs

    def _beam_search(
        self,
        strings: torch.Tensor,
        types: torch.Tensor,
        max_regex_length: int,
        beam_size: int,
        temperature: float = 1.0,
    ):
        """
        Beam search that returns multiple candidates.

        Args:
            strings: Input strings tensor
            types: Type tensor (0 for positive, 1 for negative)
            max_regex_length: Maximum length of regex to generate
            beam_size: Number of beams to maintain and return as final candidates
            temperature: Temperature for probability distribution

        Returns:
            regexes: Top beam_size regexes (beam_size, max_regex_length-1)
            scores: Scores for each candidate (beam_size,)
        """
        device = self.device
        vocab_size = RegexVocabulary.get_vocabulary_size()
        eos_token_index = RegexVocabulary.eos_token_index

        # Encode input once
        string_embedding, set_embedding = self.encode(strings, types)

        # Initialize beams: start with <sos> token
        beams = torch.full((beam_size, max_regex_length), self.output_pad_token_index, dtype=torch.long).to(device)
        beams[:, 0] = RegexVocabulary.sos_token_index
        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = float('-inf')  # Only first beam is active initially

        # Expand encoder outputs for all beams
        string_embedding_expanded = string_embedding.expand(beam_size, -1, -1)
        set_embedding_expanded = set_embedding.expand(beam_size, -1, -1)

        target_mask = nn.Transformer.generate_square_subsequent_mask(max_regex_length - 1).to(device)
        finished_beams = []  # List of (score, sequence)

        for t in range(max_regex_length - 1):
            # Get current active beams
            active_beams = beam_scores > float('-inf')
            n_active = active_beams.sum().item()

            if n_active == 0:
                break

            # Decode current step
            decoder_inputs = beams[:, : t + 1]
            target_key_padding_mask = torch.eq(decoder_inputs, self.output_pad_token_index).to(device)
            decoder_embedding = self.regex_embedding(decoder_inputs)
            decoder_embedding = decoder_embedding * math.sqrt(self.hidden_size) + self.positional_encoding(decoder_embedding)

            output = self.set_decoder(
                decoder_embedding,
                set_embedding_expanded,
                tgt_key_padding_mask=target_key_padding_mask,
                tgt_is_causal=True,
                tgt_mask=target_mask[: t + 1, : t + 1],
            )
            output = self.string_decoder(
                output,
                string_embedding_expanded,
                tgt_key_padding_mask=target_key_padding_mask,
                tgt_is_causal=True,
                tgt_mask=target_mask[: t + 1, : t + 1],
            )

            logits = self.out(output[:, -1])  # beam_size, vocab_size
            log_probs = F.log_softmax(logits / temperature, dim=-1)

            # Compute scores for all possible next tokens
            # beam_scores: (beam_size,) -> (beam_size, 1)
            # log_probs: (beam_size, vocab_size)
            next_scores = beam_scores.unsqueeze(1) + log_probs  # beam_size, vocab_size
            next_scores = next_scores.view(-1)  # beam_size * vocab_size

            # Get top beam_size candidates
            top_scores, top_indices = torch.topk(next_scores, min(beam_size * 2, len(next_scores)))

            # Determine which beam and which token each candidate came from
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Build new beams
            new_beams = []
            new_scores = []
            new_beam_indices = []

            for i in range(len(top_scores)):
                beam_idx = beam_indices[i].item()
                token_idx = token_indices[i].item()
                score = top_scores[i].item()

                # Create new beam
                new_beam = beams[beam_idx].clone()
                new_beam[t + 1] = token_idx

                if token_idx == eos_token_index:
                    # Finished beam - normalize by length
                    length = t + 1
                    normalized_score = score / length
                    finished_beams.append((normalized_score, new_beam.clone()))
                else:
                    # Active beam
                    if len(new_beams) < beam_size:
                        new_beams.append(new_beam)
                        new_scores.append(score)
                        new_beam_indices.append(beam_idx)

            # If all beams finished, stop
            if not new_beams:
                break

            # Update beams
            beams = torch.stack(new_beams)
            beam_scores = torch.tensor(new_scores, device=device)
            beam_indices_tensor = torch.tensor(new_beam_indices, device=device)

            # Reorder encoder outputs according to which beams survived
            string_embedding_expanded = string_embedding_expanded[beam_indices_tensor]
            set_embedding_expanded = set_embedding_expanded[beam_indices_tensor]

            # Pad beams if needed
            if len(beams) < beam_size:
                padding_size = beam_size - len(beams)
                pad_beams = torch.full((padding_size, max_regex_length), self.output_pad_token_index, dtype=torch.long, device=device)
                beams = torch.cat([beams, pad_beams])
                beam_scores = torch.cat([beam_scores, torch.full((padding_size,), float('-inf'), device=device)])

                # Pad encoder outputs
                string_embedding_expanded = torch.cat([string_embedding_expanded, string_embedding.expand(padding_size, -1, -1)])
                set_embedding_expanded = torch.cat([set_embedding_expanded, set_embedding.expand(padding_size, -1, -1)])

        # Add remaining active beams to finished (normalize by actual length)
        for i, score in enumerate(beam_scores):
            if score > float('-inf'):
                # Find actual length (first padding token)
                seq = beams[i]
                length = (seq != self.output_pad_token_index).sum().item() - 1  # -1 for <sos>
                normalized_score = score / max(length, 1)
                finished_beams.append((normalized_score, beams[i].clone()))

        # Sort by score and return top beam_size candidates
        finished_beams.sort(key=lambda x: x[0], reverse=True)

        # Get top beam_size candidates
        n_return = min(beam_size, len(finished_beams))
        result_beams = torch.stack([beam for _, beam in finished_beams[:n_return]])
        result_scores = torch.tensor([score for score, _ in finished_beams[:n_return]], device=device)

        # If we don't have enough, pad with the best ones repeated or empty sequences
        if n_return < beam_size:
            padding_size = beam_size - n_return
            if n_return > 0:
                # Repeat best beam
                pad_beams = result_beams[-1:].repeat(padding_size, 1)
                pad_scores = result_scores[-1:].repeat(padding_size)
            else:
                # Return empty sequences
                pad_beams = torch.full((padding_size, max_regex_length), self.output_pad_token_index, dtype=torch.long, device=device)
                pad_beams[:, 0] = RegexVocabulary.sos_token_index
                pad_beams[:, 1] = eos_token_index
                pad_scores = torch.full((padding_size,), float('-inf'), device=device)
            result_beams = torch.cat([result_beams, pad_beams])
            result_scores = torch.cat([result_scores, pad_scores])

        # Remove <sos> token and return
        return result_beams[:, 1:], result_scores

    def _reset_parameters(self):
        """
        All layers in the TransformerEncoder / TransformerDecoder are initialized with the same parameters.
        It is recommended to manually initialize the layers after creating the TransformerEncoder / TransformerDecoder instance.
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def _sample(self, logits, temperature=1.0, k=0, p=0.0):
        if temperature > 0.0:
            logits /= temperature
        if k > 0:
            topk_values, _ = torch.topk(logits, k)  # batch_size, k
            mask = logits < topk_values[:, -1].unsqueeze(1)
            logits = logits.masked_fill(mask, float('-inf'))
        if p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)  # batch_size, vocab_size
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        return next_token
