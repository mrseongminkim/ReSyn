import pickle

from utils.engine import EngineException, fullmatch
from utils.exceptions import SynthesisFailure


class GPTOssSynthesizer:
    def __init__(self):
        self.results_map = self._load_results()

    def _load_results(self):
        with open('logs/gpt-oss-vllm/gpt_oss_results.pkl', 'rb') as f:
            mapping = pickle.load(f)
        return mapping

    def synthesize(self, positive_strings, negative_strings, id):
        regex = self.results_map[id]['answer']
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
