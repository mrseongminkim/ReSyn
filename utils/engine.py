import re2

options = re2.Options()
options.log_errors = False
options.dot_nl = True

EngineException = re2.error


def escape(pattern):
    return re2.escape(pattern)


def fullmatch(pattern, string):
    return re2.fullmatch(pattern, string, options)


def compile(pattern):
    return re2.compile(pattern, options)
