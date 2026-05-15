import math
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

from config import CheckPointConfig
from utils.engine import fullmatch
from utils.exceptions import SynthesisFailure, get_smallest_named_character_class
from utils.normalizer import NormalizeException, normalize

from .dataset import PartitionerVocabulary, RegexVocabulary, SegmenterVocabulary, StringVocabulary
from .model import Partitioner, Router, Segmenter, Set2Regex


def string_to_segmenter_indices(strings):
    strings_indices = []
    empty_string_token = StringVocabulary.empty_string_token
    empty_set_index = StringVocabulary.empty_set_token_index
    for string in strings:
        tokens = list(string) if string else [empty_string_token]
        strings_indices.append(StringVocabulary.convert_tokens_to_indices(tokens))
    strings_indices = strings_indices if strings_indices else [[empty_set_index]]
    string_separater = StringVocabulary.empty_set_token_index
    input_ids = []
    for s in strings_indices:
        if input_ids:
            input_ids.append(string_separater)
        input_ids.extend(s)
    return torch.tensor(input_ids).view((1, -1))


def string_to_splitter_indices(strings, skip_padding=False):
    strings_indices = []
    empty_string_token = StringVocabulary.empty_string_token
    empty_set_index = StringVocabulary.empty_set_token_index
    for string in strings:
        tokens = list(string) if string else [empty_string_token]
        strings_indices.append(StringVocabulary.convert_tokens_to_indices(tokens))
    strings_indices = strings_indices if strings_indices else [[empty_set_index]]
    if not skip_padding:
        strings_indices = (
            pad_sequence([torch.tensor(s) for s in strings_indices], batch_first=True, padding_value=StringVocabulary.pad_token_index)
            .view(1, len(strings), -1)
            .cuda()
        )
    return strings_indices


class PartitionerServer:
    def __init__(self):
        model = Partitioner().cuda()
        model.load_state_dict(torch.load(f'checkpoints/partitioner/{CheckPointConfig.partitioner}', weights_only=True))
        model.eval()
        self.model = torch.compile(model)

    @torch.no_grad()
    def _get_labels(self, strings) -> list[int]:
        pos = string_to_splitter_indices(strings)
        labels, _ = self.model.predict(pos)
        labels = PartitionerVocabulary.convert_indices_to_tokens(labels[0])
        labels = [ord(label) for label in labels]
        return labels

    def split(self, positive_strings: list[str]) -> list[list[str]]:
        labels = self._get_labels(positive_strings)
        clustered_strings = defaultdict(set)
        for string, label in zip(positive_strings, labels):
            clustered_strings[label].add(string)
        splited_strings = []
        for gid in sorted(clustered_strings.keys()):
            splited_strings.append(list(clustered_strings[gid]))
        return splited_strings


class SegmenterServer:
    def __init__(self):
        model = Segmenter().cuda()
        checkpoint = torch.load(f'checkpoints/segmenter/{CheckPointConfig.segmenter}', weights_only=False)
        state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
        model.load_state_dict(state_dict)
        model.eval()
        self.model = torch.compile(model)

    @torch.no_grad()
    def _generate(self, curr_token, memory, cache, step):
        dec_embeds = self.model.dec_emb(curr_token)
        pe = self.model.positional_encoding(dec_embeds, step)
        dec_embeds = dec_embeds * math.sqrt(self.model.d_model) + pe
        for i, layer in enumerate(self.model.decoder_layers):
            dec_embeds, new_kv = layer(dec_embeds, memory, past_kv=cache[i], is_inference=True)
            cache[i] = new_kv
        dec_embeds = self.model.dec_final_norm(dec_embeds)
        logits = self.model.fc_out(dec_embeds)
        return logits, cache

    @torch.no_grad()
    def split(self, positive_strings: list[str]) -> list[list[str]]:
        input_ids = string_to_segmenter_indices(positive_strings).cuda()
        input_embeds = self.model.enc_emb(input_ids)
        input_embeds = input_embeds * math.sqrt(self.model.d_model) + self.model.positional_encoding(input_embeds)
        memory = self.model.encoder(input_embeds)
        curr_token = torch.full((1, 1), SegmenterVocabulary.sos_token_index, dtype=torch.long, device=input_ids.device)
        cache = [None] * len(self.model.decoder_layers)
        gid_to_set = defaultdict(set)
        gid_counter = defaultdict(int)
        step = 0
        add_empty_string = False
        for positive_string in positive_strings:
            gid_to_substrings = defaultdict(str)
            if not positive_string:
                add_empty_string = True
                _, cache = self._generate(curr_token, memory, cache, step)
                # force generate empty string token
                empty_string_token_id = SegmenterVocabulary.empty_string_token_index
                curr_token = torch.full((1, 1), empty_string_token_id, dtype=torch.long, device=curr_token.device)
                step += 1
            else:
                for c in positive_string:
                    logits, cache = self._generate(curr_token, memory, cache, step)
                    # add constraints
                    logits[:, -1, SegmenterVocabulary.empty_string_token_index] = float('-inf')
                    logits[:, -1, SegmenterVocabulary.sos_token_index] = float('-inf')
                    logits[:, -1, SegmenterVocabulary.pad_token_index] = float('-inf')
                    curr_token_id = curr_token.item()
                    if curr_token_id != SegmenterVocabulary.sos_token_index:
                        logits[:, -1, :curr_token_id] = float('-inf')
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    curr_token = next_token
                    step += 1
                    token_id = next_token.item()
                    gid = SegmenterVocabulary.index_to_token[token_id]
                    gid_to_substrings[gid] += c
            _, cache = self._generate(curr_token, memory, cache, step)
            # force generate seperator token
            seperator_token_id = SegmenterVocabulary.sos_token_index
            next_token = torch.full((1, 1), seperator_token_id, dtype=torch.long, device=curr_token.device)
            curr_token = next_token
            step += 1
            for gid, substring in gid_to_substrings.items():
                gid_to_set[gid].add(substring)
                gid_counter[gid] += 1
        if add_empty_string:
            for gid in gid_to_set.keys():
                gid_to_set[gid].add('')
        else:
            for gid in gid_counter.keys():
                if gid_counter[gid] != len(positive_strings):
                    gid_to_set[gid].add('')
        result = []
        for gid in sorted(gid_to_set.keys()):
            result.append(list(sorted(gid_to_set[gid])))
        return result


class RouterServer:
    def __init__(self):
        model = Router().cuda()
        model.load_state_dict(torch.load(f'checkpoints/router/{CheckPointConfig.router}', weights_only=True))
        model.eval()
        self.model = torch.compile(model)
        self.label = ['concat', 'union', 'no-op']

    @torch.no_grad()
    def classify(self, positive_strings: list[str]) -> str:
        pos = string_to_splitter_indices(positive_strings)
        log_probs = self.model(pos)
        predictions = log_probs.argmax(dim=-1)
        return self.label[predictions.item()]


class Set2RegexServer:
    def __init__(self, do_sample=False, debug=False, beam_size=0):
        model = Set2Regex().cuda()
        model.load_state_dict(torch.load(f'checkpoints/set2regex/{CheckPointConfig.set2regex}', weights_only=True, map_location='cpu'))
        model.cuda()
        model.eval()
        self.model = torch.compile(model)
        self.gen_config = {
            'p': 0.9 if do_sample else 0.0,
            'k': 50 if do_sample else 0,
            'temperature': 1.0,
            'n_candidates': 500 if do_sample else 1,
        }
        self.debug = debug
        self.beam_size = beam_size
        self.fallback = False

    @torch.no_grad()
    def synthesize_all(self, positive_strings: list[str], negative_strings: list[str]):
        pos = string_to_splitter_indices(positive_strings, skip_padding=True)
        neg = string_to_splitter_indices(negative_strings, skip_padding=True)
        strings = pos + neg
        n_strings = len(strings)
        strings = (
            pad_sequence([torch.tensor(s) for s in strings], batch_first=True, padding_value=StringVocabulary.pad_token_index)
            .view(1, n_strings, -1)
            .cuda()
        )
        types = [0] * len(pos) + [1] * len(neg)
        types = torch.tensor(types).view(1, n_strings).cuda()
        regexes, _ = self.model.predict(strings, types, **self.gen_config, beam_size=self.beam_size)
        if self.debug:
            _regexes = []
            for regex in regexes:
                regex = RegexVocabulary.convert_indices_to_tokens(regex)
                try:
                    regex = regex[: regex.index('<eos>')]
                except ValueError:
                    continue
                regex = ''.join(regex)
                _regexes.append(regex)
            print(set(_regexes))
        for i, regex in enumerate(regexes):  # regexes are already sorted by their log-probabilities
            regex = RegexVocabulary.convert_indices_to_tokens(regex)
            try:
                regex = regex[: regex.index('<eos>')]
            except ValueError:
                continue
            regex = ''.join(regex)
            try:
                ast = normalize(regex)
                regex = ast.get_regex()
                needs_question_mark = False
                for positive_string in positive_strings:
                    if not positive_string and not fullmatch(regex, positive_string):
                        needs_question_mark = True
                        break
                if needs_question_mark:
                    regex = f'({regex})?'
            except NormalizeException:
                continue
            if any(not fullmatch(regex, s) for s in positive_strings) or any(fullmatch(regex, s) for s in negative_strings):
                continue
            yield regex
        if self.fallback:
            yield get_smallest_named_character_class(positive_strings, negative_strings)

    def synthesize(self, positive_strings: list[str], negative_strings: list[str]) -> str:
        for regex in self.synthesize_all(positive_strings, negative_strings):
            return regex
        raise SynthesisFailure
