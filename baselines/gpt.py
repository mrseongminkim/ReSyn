import json

from utils.engine import fullmatch, EngineException
from utils.exceptions import SynthesisFailure


class GPTSynthesizer:
    def __init__(self):
        batch_result_path = 'logs/gpt/gpt5_2026_01_08_results.jsonl'
        self.results_map = self._load_results(batch_result_path)

    def _load_results(self, path):
        mapping = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                case_id = int(data.get('custom_id'))
                if data.get('response', {}).get('status_code') == 200:
                    body = data['response']['body']
                    regex = body['choices'][0]['message']['content'].strip()
                mapping[case_id] = regex
        return mapping

    def synthesize(self, positive_strings, negative_strings, id):
        regex = self.results_map[id]
        try:
            for positive_string in positive_strings:
                if not fullmatch(regex, positive_string):
                    raise SynthesisFailure
            for negative_string in negative_strings:
                if fullmatch(regex, negative_string):
                    raise SynthesisFailure
        except EngineException:
            raise SynthesisFailure
        return regex
