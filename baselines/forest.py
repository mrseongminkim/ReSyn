from baselines.FOREST.forest.synthesizer.multitree_synthesizer import MultiTreeSynthesizer

splitter = MultiTreeSynthesizer(
    valid_examples=[None], invalid_examples=[], captured=None, condition_invalid=[], main_dsl=None, ground_truth=None
)


def forest_split(positive_strings):
    positive_strings = [[s] for s in positive_strings]
    splitter.valid = positive_strings
    splitter.invalid = []
    try:
        valid, _ = splitter.split_examples()
    except IndexError:
        valid = None
    if valid is None:
        return [positive_strings]
    columns = list(zip(*valid))
    columns = [list(set(col)) for col in columns]
    return columns
