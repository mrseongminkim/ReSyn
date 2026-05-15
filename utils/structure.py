import re._parser as parser
import string
from abc import ABC, abstractmethod
from itertools import groupby
from operator import itemgetter

from . import engine


class Node(ABC):
    def __init__(self):
        self.type = self.__class__.__name__.lower()

    @abstractmethod
    def get_regex(self) -> str:
        pass

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.get_regex() < other.get_regex()

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.get_regex() == other.get_regex()

    def __hash__(self):
        return hash(self.get_regex())

    @abstractmethod
    def get_signature(self):
        pass

    @abstractmethod
    def get_depth(self) -> int:
        pass

    @abstractmethod
    def count_descendants(self) -> int:
        pass

    @abstractmethod
    def count_literals(self) -> int:
        """Count the number of literal nodes."""
        pass

    @abstractmethod
    def count_alternations(self) -> int:
        """Count the number of union (|) operations."""
        pass

    @abstractmethod
    def get_character_classes(self) -> set[str]:
        """Get all character classes used in the regex."""
        pass

    @abstractmethod
    def get_repetitions(self) -> set[str]:
        """Get all unique repetition quantifiers used in the regex."""
        pass

    def get_node_count(self) -> int:
        """Return the total number of nodes including this node."""
        return 1 + self.count_descendants()

    def get_operator_density(self) -> float:
        """Calculate operator density: (total_nodes - literals) / total_nodes."""
        total = self.get_node_count()
        literals = self.count_literals()
        return (total - literals) / total if total > 0 else 0.0


class Empty(Node):
    def __init__(self):
        super().__init__()

    def get_regex(self):
        return ''

    def get_signature(self):
        return '<EMPTY>'

    def get_depth(self):
        return 1

    def count_descendants(self):
        return 0

    def count_literals(self):
        return 0

    def count_alternations(self):
        return 0

    def get_character_classes(self):
        return set()

    def get_repetitions(self):
        return set()


class Literal(Node):
    def __init__(self, literal):
        super().__init__()
        self.literal = literal

    def get_regex(self):
        return engine.escape(self.literal)

    def get_signature(self):
        return '<LITERAL>'

    def get_depth(self):
        return 1

    def count_descendants(self):
        return 0

    def count_literals(self):
        return 1

    def count_alternations(self):
        return 0

    def get_character_classes(self):
        return set()

    def get_repetitions(self):
        return set()


class CharacterClass(Node):
    named_character_classes = {
        frozenset(string.whitespace.replace('\x0b', '')): r'\s',
        frozenset(string.digits): r'\d',
        frozenset(set(string.printable).difference(set(string.digits + string.ascii_letters + '_'))): r'\W',
        frozenset(set(string.digits + string.ascii_letters + '_')): r'\w',
        frozenset(set(string.printable).difference(set(string.digits))): r'\D',
        frozenset(set(string.printable).difference(set(string.whitespace))): r'\S',
        frozenset(string.printable): r'.',
    }

    def __init__(self, characters: set[str]):
        super().__init__()
        self.characters = characters

    def get_regex(self):
        key = frozenset(self.characters)
        if key in self.named_character_classes:
            return self.named_character_classes[key]
        ords = sorted(ord(c) for c in self.characters)
        ranges = []
        for _, group in groupby(enumerate(ords), lambda t: t[1] - t[0]):
            block = list(map(itemgetter(1), group))
            if len(block) >= 3:
                ranges.append(f'{engine.escape(chr(block[0]))}-{engine.escape(chr(block[-1]))}')
            else:
                ranges.extend(engine.escape(chr(o)) for o in block)
        return f'[{"".join(ranges)}]'

    def get_signature(self):
        return f'<CHAR_CLASS>({sorted(self.characters)})'

    def get_depth(self):
        return 1

    def count_descendants(self):
        return 0

    def count_literals(self):
        return 0

    def count_alternations(self):
        return 0

    def get_character_classes(self):
        return {self.get_regex()}

    def get_repetitions(self):
        return set()


class Repetition(Node):
    def __init__(self, child: Node, min_repeat, max_repeat):
        super().__init__()
        self.child = child
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat

    def get_regex(self):
        subregex = self.child.get_regex()
        if self.child.type in {'union', 'concat', 'repetition'}:
            subregex = f'({subregex})'
        if self.min_repeat == 0 and self.max_repeat == int(parser.MAXREPEAT):
            quantifier = '*'
        elif self.min_repeat == 1 and self.max_repeat == int(parser.MAXREPEAT):
            quantifier = '+'
        elif self.min_repeat == 0 and self.max_repeat == 1:
            quantifier = '?'
        elif self.min_repeat == self.max_repeat:
            quantifier = f'{{{self.min_repeat}}}'
        else:
            quantifier = f'{{{self.min_repeat},{self.max_repeat}}}'
        return f'{subregex}{quantifier}'

    def get_signature(self):
        child = self.child.get_signature()
        return f'<REPEAT>({child},{self.min_repeat},{self.max_repeat})'

    def get_depth(self):
        return 1 + self.child.get_depth()

    def count_descendants(self):
        return 1 + self.child.count_descendants()

    def count_literals(self):
        return self.child.count_literals()

    def count_alternations(self):
        return self.child.count_alternations()

    def get_character_classes(self):
        return self.child.get_character_classes()

    def get_repetitions(self):
        quantifier = ''
        if self.min_repeat == 0 and self.max_repeat == int(parser.MAXREPEAT):
            quantifier = '*'
        elif self.min_repeat == 1 and self.max_repeat == int(parser.MAXREPEAT):
            quantifier = '+'
        elif self.min_repeat == 0 and self.max_repeat == 1:
            quantifier = '?'
        elif self.min_repeat == self.max_repeat:
            quantifier = f'{{{self.min_repeat}}}'
        else:
            quantifier = f'{{{self.min_repeat},{self.max_repeat}}}'
        result = {quantifier}
        result.update(self.child.get_repetitions())
        return result


class Concat(Node):
    def __init__(self, children: list[Node]):
        super().__init__()
        self.children = children

    def get_regex(self):
        regex = []
        for child in self.children:
            subregex = child.get_regex()
            if child.type == 'union':
                subregex = f'({subregex})'
            regex.append(subregex)
        return ''.join(regex)

    def get_signature(self):
        children = [child.get_signature() for child in self.children]
        return f'<CONCAT>({"".join(children)})'

    def get_depth(self):
        return 1 + max(child.get_depth() for child in self.children)

    def count_descendants(self):
        return len(self.children) + sum(child.count_descendants() for child in self.children)

    def count_literals(self):
        return sum(child.count_literals() for child in self.children)

    def count_alternations(self):
        return sum(child.count_alternations() for child in self.children)

    def get_character_classes(self):
        result = set()
        for child in self.children:
            result.update(child.get_character_classes())
        return result

    def get_repetitions(self):
        result = set()
        for child in self.children:
            result.update(child.get_repetitions())
        return result


class Union(Node):
    def __init__(self, children: list[Node]):
        super().__init__()
        self.children = children

    def get_regex(self):
        regex = []
        for child in sorted(self.children):
            subregex = child.get_regex()
            regex.append(subregex)
        return '|'.join(regex)

    def get_signature(self):
        children = sorted([child.get_signature() for child in self.children])
        return f'<UNION>({"".join(children)})'

    def get_depth(self):
        return 1 + max(child.get_depth() for child in self.children)

    def count_descendants(self):
        return len(self.children) + sum(child.count_descendants() for child in self.children)

    def count_literals(self):
        return sum(child.count_literals() for child in self.children)

    def count_alternations(self):
        return 1 + sum(child.count_alternations() for child in self.children)

    def get_character_classes(self):
        result = set()
        for child in self.children:
            result.update(child.get_character_classes())
        return result

    def get_repetitions(self):
        result = set()
        for child in self.children:
            result.update(child.get_repetitions())
        return result
