import re
import re._parser as parser
import string

from .structure import CharacterClass, Concat, Empty, Literal, Node, Repetition, Union


class BuilderException(Exception):
    pass


class Builder:
    max_repeat_limit: int = 10

    def build(self, regex, is_anonymized: bool = False) -> Concat:
        self.is_anonymized = is_anonymized
        nodes = []
        try:
            tokens = parser.parse(regex)
        except (re.error, OverflowError):
            raise BuilderException('Failed to parse regex')
        for opcode, arguments in tokens:
            node = self.handle(opcode, arguments)
            nodes.append(node)
        return Concat(nodes)

    def handle(self, opcode, arguments) -> Node:
        method_name = f'handle_{str(opcode).lower()}'
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(arguments)
        elif opcode in {
            parser.ASSERT,
            parser.ASSERT_NOT,
            parser.GROUPREF,
            parser.GROUPREF_EXISTS,
        }:
            raise BuilderException('Lookaround assertions and backreferences are not supported')
        else:
            raise NotImplementedError(f'Handler for opcode {opcode} is not implemented')

    def handle_literal(self, ordinal: int) -> Literal:
        character = chr(ordinal)
        if character not in string.printable and not self.is_anonymized:
            raise BuilderException(f"Non-printable literal '{character}' is not supported")
        return Literal(character)

    def handle_not_literal(self, ordinal: int) -> CharacterClass:
        character = chr(ordinal)
        if character not in string.printable:
            raise BuilderException(f"Non-printable literal '{character}' is not supported")
        characters = set(string.printable.replace(character, ''))
        if not characters:
            raise BuilderException('Empty character class is not supported')
        return CharacterClass(characters)

    def handle_any(self, _: None) -> CharacterClass:
        return CharacterClass(set(string.printable))

    def handle_category(self, category) -> CharacterClass:
        category_map = {
            parser.CATEGORY_SPACE: set(string.whitespace),
            parser.CATEGORY_NOT_SPACE: set(string.printable).difference(set(string.whitespace)),
            parser.CATEGORY_DIGIT: set(string.digits),
            parser.CATEGORY_NOT_DIGIT: set(string.printable).difference(set(string.digits)),
            parser.CATEGORY_WORD: set(string.ascii_letters + string.digits + '_'),
            parser.CATEGORY_NOT_WORD: set(string.printable).difference(set(string.ascii_letters + string.digits + '_')),
        }
        return CharacterClass(category_map[category])

    def handle_max_repeat(self, args) -> Repetition:
        return self.handle_repeat(args)

    def handle_min_repeat(self, args) -> Repetition:
        return self.handle_repeat(args)

    def handle_possessive_repeat(self, args) -> Repetition:
        return self.handle_repeat(args)

    def handle_repeat(self, args) -> Repetition:
        min_repeat, max_repeat, sub_token = args
        if max_repeat == parser.MAXREPEAT:
            max_repeat = int(parser.MAXREPEAT)
            if min_repeat not in {0, 1}:
                max_repeat = self.max_repeat_limit
                if min_repeat > max_repeat:
                    min_repeat = max_repeat
        elif max_repeat > self.max_repeat_limit:
            max_repeat = self.max_repeat_limit
            if min_repeat > max_repeat:
                min_repeat = max_repeat
        if not sub_token:
            return Empty()
        opcode, arguments = sub_token[0]
        node = self.handle(opcode, arguments)
        return Repetition(node, min_repeat, max_repeat)

    def handle_range(self, args) -> CharacterClass:
        min_value, max_value = args
        characters = set(chr(i) for i in range(min_value, max_value + 1))
        if not characters:
            raise BuilderException('Empty character class is not supported')
        elif characters.difference(set(string.printable)) and not self.is_anonymized:
            raise BuilderException(f"Non-printable range '{chr(min_value)}-{chr(max_value)}' is not supported")
        return CharacterClass(characters)

    def handle_branch(self, sub_tokens) -> Union:
        sub_tokens = sub_tokens[1]
        nodes = []
        for group in sub_tokens:
            sub_nodes = []
            for opcode, arguments in group:
                node = self.handle(opcode, arguments)
                sub_nodes.append(node)
            nodes.append(Concat(sub_nodes))
        return Union(nodes)

    def handle_at(self, code) -> Empty:
        return Empty()

    def handle_atomic_group(self, sub_tokens) -> Concat:
        sub_nodes = []
        for opcode, arguments in sub_tokens:
            node = self.handle(opcode, arguments)
            sub_nodes.append(node)
        return Concat(sub_nodes)

    def handle_subpattern(self, sub_tokens) -> Concat:
        sub_nodes = []
        sub_tokens = sub_tokens[3]
        for opcode, arguments in sub_tokens:
            node = self.handle(opcode, arguments)
            sub_nodes.append(node)
        return Concat(sub_nodes)

    def handle_in(self, sub_tokens) -> CharacterClass:
        characters = set()
        negate = False
        for opcode, arguments in sub_tokens:
            if opcode == parser.NEGATE:
                negate = True
            else:
                node = self.handle(opcode, arguments)
                if isinstance(node, CharacterClass):
                    characters.update(node.characters)
                elif isinstance(node, Literal):
                    characters.add(node.literal)
                else:
                    raise BuilderException(f'Unsupported node type in character class: {type(node)}')
        if negate:
            characters = set(string.printable).difference(characters)
        if not characters:
            raise BuilderException('Empty character class is not supported')
        return CharacterClass(characters)
