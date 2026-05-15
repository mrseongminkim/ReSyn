from utils.engine import escape, fullmatch

class TrivialSynthesizer:
    def synthesize(self, positive_strings: list[str], negative_strings: list[str]) -> str:
        regexes = []
        for positive_string in positive_strings:
            sub_regex = escape(positive_string)
            regexes.append(sub_regex)
        regex = '|'.join(regexes)
        for positive_string in positive_strings:
            if not fullmatch(regex, positive_string):
                breakpoint()
        for negative_string in negative_strings:
            if fullmatch(regex, negative_string):
                breakpoint()
        return regex
