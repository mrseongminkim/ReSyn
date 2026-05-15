from .engine import fullmatch


class SynthesisFailure(Exception):
    pass


def get_smallest_named_character_class(positive_strings: list[str], negative_strings: list[str]) -> str:
    for named_character_class in (r'\d', r'[a-z]', r'[A-Z]', r'[a-zA-Z]', r'[0-9a-fA-F]', r'\w', r'\s', r'\D', r'\W', r'\S', r'.'):
        for quantifier in ('+', '*'):
            regex = named_character_class + quantifier
            if all(fullmatch(regex, s) for s in positive_strings) and all(not fullmatch(regex, s) for s in negative_strings):
                return regex
    raise SynthesisFailure
