from .structure import CharacterClass, Concat, Empty, Literal, Node, Repetition, Union


class OptimizerException(Exception):
    pass


class Optimizer:
    def optimize(self, node: Node, valid_empty_token: bool = False) -> Node:
        self.valid_empty_token = valid_empty_token
        while True:
            self.dirty = False
            node = self._optimize(node)
            if not self.dirty:
                break
        if node.type == 'empty':
            raise OptimizerException('Optimized to empty regex')
        return node

    def _optimize(self, node: Node) -> Node:
        if node.type in {'concat', 'union'}:
            optimized_children = []
            for child in node.children:
                optimized_child = self._optimize(child)
                optimized_children.append(optimized_child)
            node.children = optimized_children
        elif node.type == 'repetition':
            node.child = self._optimize(node.child)
        elif node.type in {'literal', 'empty'}:
            return node
        method_name = f'optimize_{node.type}'
        method = getattr(self, method_name)
        return method(node)

    def optimize_characterclass(self, node: CharacterClass) -> Node:
        if len(node.characters) == 0:
            raise OptimizerException('CharacterClass with empty characters is not allowed')
        elif len(node.characters) == 1:
            self.dirty = True
            return Literal(next(iter(node.characters)))
        return node

    def optimize_repetition(self, node: Repetition) -> Node:
        if node.min_repeat == 0 and node.max_repeat == 0:
            self.dirty = True
            return Empty()
        elif node.min_repeat == 1 and node.max_repeat == 1:
            self.dirty = True
            return node.child
        elif node.min_repeat == node.max_repeat and isinstance(node.child, Literal):
            combined_literal = node.child.literal * node.min_repeat
            self.dirty = True
            return Literal(combined_literal)
        if isinstance(node.child, Empty):
            self.dirty = True
            return node.child
        return node

    def optimize_concat(self, node: Concat) -> Node:
        new_children = []
        for child in node.children:
            if isinstance(child, Concat):
                new_children.extend(child.children)
                self.dirty = True
            elif isinstance(child, Empty):
                self.dirty = True
            elif isinstance(child, Literal):
                if new_children and isinstance(new_children[-1], Literal):
                    combined_literal = new_children[-1].literal + child.literal
                    new_children[-1] = Literal(combined_literal)
                    self.dirty = True
                else:
                    new_children.append(child)
            else:
                new_children.append(child)
        node.children = new_children
        if len(node.children) == 0:
            self.dirty = True
            return Empty()
        elif len(node.children) == 1:
            self.dirty = True
            return node.children[0]
        return node

    def optimize_union(self, node: Union) -> Node:
        new_children = []
        for child in node.children:
            if isinstance(child, Union):
                new_children.extend(child.children)
                self.dirty = True
            elif isinstance(child, Empty) and not self.valid_empty_token:
                self.dirty = True
            else:
                new_children.append(child)
        unique_children = []
        seen_regexes = set()
        for child in new_children:
            regex = child.get_regex()
            if regex not in seen_regexes:
                seen_regexes.add(regex)
                unique_children.append(child)
        node.children = sorted(unique_children, key=lambda x: x.get_regex())
        if len(node.children) == 0:
            self.dirty = True
            return Empty()
        elif len(node.children) == 1:
            self.dirty = True
            return node.children[0]
        return node
