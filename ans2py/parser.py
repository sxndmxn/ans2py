"""APL Parser - Build AST from tokens."""

from dataclasses import dataclass, field
from typing import Union

from .lexer import Token, TokenType, tokenize


# AST Node Types
@dataclass
class Scalar:
    """A scalar value (number or string)."""
    value: Union[int, float, complex, str]


@dataclass
class Array:
    """An array literal."""
    elements: list


@dataclass
class Name:
    """A variable or function name."""
    name: str


@dataclass
class Monad:
    """Monadic function application: fn arg"""
    fn: str
    right: 'Node'


@dataclass
class Dyad:
    """Dyadic function application: left fn right"""
    fn: str
    left: 'Node'
    right: 'Node'


@dataclass
class MonadicOp:
    """Monadic operator: fn op arg (e.g., +/ for reduce)"""
    op: str
    fn: str
    right: 'Node'


@dataclass
class DyadicOp:
    """Dyadic operator: left fn1.fn2 right (e.g., inner product)"""
    op: str
    fn1: str
    fn2: str
    left: 'Node'
    right: 'Node'


@dataclass
class Assignment:
    """Variable assignment: name ← value"""
    name: str
    value: 'Node'


@dataclass
class Index:
    """Array indexing: array[indices]"""
    array: 'Node'
    indices: list


@dataclass
class Function:
    """User-defined function."""
    name: str
    result: str = ''
    left_arg: str = ''
    right_arg: str = ''
    locals: list = field(default_factory=list)
    body: list = field(default_factory=list)


@dataclass
class Dfn:
    """Direct function (dfn): { body }"""
    body: list


@dataclass
class Guard:
    """Guard expression in dfn: condition : result"""
    condition: 'Node'
    result: 'Node'


# Type alias for all node types
Node = Union[Scalar, Array, Name, Monad, Dyad, MonadicOp, DyadicOp,
             Assignment, Index, Function, Dfn, Guard]


class Parser:
    """Parse APL tokens into AST."""

    PRIMITIVES = set('+-×÷*⍟!○|⌈⌊⊥⊤⊣⊢<≤=≥>≠≡≢∧∨⍲⍱~⍳⍴⍪,⌽⊖⍉↑↓⊂⊃∊⍷⍸⌷⍋⍒?⍎⍕⌹∪∩')
    OPERATORS = set('/\\¨⍨⍣⍤⍥∘.@⌸⌺⍠')

    def __init__(self, tokens: list[Token]):
        self.tokens = [t for t in tokens if t.type not in (TokenType.COMMENT, TokenType.NEWLINE)]
        self.pos = 0

    def parse(self) -> list[Node]:
        """Parse all statements."""
        statements = []
        while not self._at_end():
            stmt = self._parse_statement()
            if stmt is not None:
                statements.append(stmt)
        return statements

    def _parse_statement(self) -> Node:
        """Parse a single statement."""
        if self._at_end():
            return None

        # Check for assignment
        if self._check(TokenType.NAME) and self._check_ahead(TokenType.ASSIGN, 1):
            return self._parse_assignment()

        return self._parse_expression()

    def _parse_assignment(self) -> Assignment:
        """Parse an assignment statement."""
        name_token = self._advance()
        self._advance()  # Skip ←
        value = self._parse_expression()
        return Assignment(name=name_token.value, value=value)

    def _parse_expression(self) -> Node:
        """Parse an expression (right-to-left evaluation)."""
        return self._parse_array_or_function()

    def _parse_array_or_function(self) -> Node:
        """Parse array literal or function application."""
        # Collect all items that could be part of an array or expression
        raw_items = []

        while not self._at_end() and not self._check(TokenType.RPAREN) and not self._check(TokenType.RBRACKET):
            if self._check(TokenType.SEPARATOR):
                self._advance()
                break

            item = self._parse_primary()
            if item is None:
                break
            raw_items.append(item)

        if not raw_items:
            return None

        # Pre-process: combine function-operator pairs (operators bind LEFT)
        # e.g., [+, /, ⍳, 5] -> [+/, ⍳, 5]
        items = []
        i = 0
        while i < len(raw_items):
            item = raw_items[i]
            # Check if this is a function followed by an operator
            if (isinstance(item, Name) and item.name in self.PRIMITIVES and
                i + 1 < len(raw_items) and
                isinstance(raw_items[i + 1], Name) and raw_items[i + 1].name in self.OPERATORS):
                # Combine function and operator into a DerivedFunction marker
                fn = item.name
                op = raw_items[i + 1].name
                items.append(('derived', fn, op))
                i += 2
            else:
                items.append(item)
                i += 1

        if not items:
            return None

        # Right-to-left evaluation
        # Build up the expression from right to left
        result = items[-1]
        if isinstance(result, tuple) and result[0] == 'derived':
            # Derived function at end with no argument - shouldn't happen, treat as error
            _, fn, op = result
            result = Name(name=f"{fn}{op}")  # Fallback

        i = len(items) - 2
        while i >= 0:
            item = items[i]

            # Check if item is a derived function (function + operator)
            if isinstance(item, tuple) and item[0] == 'derived':
                _, fn, op = item
                # Check if immediate left is a value (not a function)
                has_value_left = self._has_value_left(items, i)
                if i > 0 and has_value_left:
                    # Dyadic derived function: collect all consecutive values to the left
                    left_items, new_i = self._collect_left_values(items, i - 1)
                    if len(left_items) == 1:
                        left = left_items[0]
                    else:
                        left = Array(elements=left_items)
                    result = DyadicOp(op=op, fn1=fn, fn2=fn, left=left, right=result)
                    i = new_i
                else:
                    # Monadic derived function: (fn op) right
                    result = MonadicOp(op=op, fn=fn, right=result)
                    i -= 1
            # Check if item is a primitive function
            elif isinstance(item, Name) and item.name in self.PRIMITIVES:
                fn = item.name
                # Check if immediate left is a value (not a function)
                has_value_left = self._has_value_left(items, i)
                if i > 0 and has_value_left:
                    # Dyadic function: collect all consecutive values to the left
                    left_items, new_i = self._collect_left_values(items, i - 1)
                    if len(left_items) == 1:
                        left = left_items[0]
                    else:
                        left = Array(elements=left_items)
                    result = Dyad(fn=fn, left=left, right=result)
                    i = new_i
                else:
                    # Monadic function
                    result = Monad(fn=fn, right=result)
                    i -= 1
            elif isinstance(item, Name) and item.name in self.OPERATORS:
                # Standalone operator without function - skip
                i -= 1
            else:
                # Regular value - could be array element or left argument
                # Check if result is a function application
                if isinstance(result, (Monad, MonadicOp)):
                    # This is left argument to the function
                    if isinstance(result, Monad):
                        result = Dyad(fn=result.fn, left=item, right=result.right)
                    elif isinstance(result, MonadicOp):
                        result = DyadicOp(op=result.op, fn1=result.fn, fn2=result.fn,
                                        left=item, right=result.right)
                else:
                    # Array literal - accumulate elements
                    if isinstance(result, Array):
                        result = Array(elements=[item] + result.elements)
                    else:
                        result = Array(elements=[item, result])
                i -= 1

        return result

    def _is_value(self, item) -> bool:
        """Check if an item is a value (not a function or operator)."""
        if isinstance(item, tuple) and item[0] == 'derived':
            return False  # It's a derived function
        if isinstance(item, Name) and (item.name in self.PRIMITIVES or item.name in self.OPERATORS):
            return False  # It's a function or operator
        return True  # It's a value

    def _has_value_left(self, items: list, index: int) -> bool:
        """Check if the immediate left neighbor is a value (non-function).
        In APL, a function is dyadic if there's a value immediately to its left."""
        if index <= 0:
            return False
        return self._is_value(items[index - 1])

    def _collect_left_values(self, items: list, start_index: int) -> tuple[list, int]:
        """Collect consecutive values starting from start_index going left.
        Returns (list of values in left-to-right order, new index to continue from).
        """
        values = []
        i = start_index
        while i >= 0 and self._is_value(items[i]):
            values.insert(0, items[i])  # Insert at front to maintain order
            i -= 1
        return values, i

    def _combine_items(self, items: list) -> Node:
        """Combine multiple items into a single node (recursively parse them)."""
        if not items:
            return None
        if len(items) == 1:
            item = items[0]
            if isinstance(item, tuple) and item[0] == 'derived':
                # Derived function without argument - return as Name for now
                _, fn, op = item
                return Name(name=f"{fn}{op}")
            return item

        # Check if any items are functions/operators - if so, need to recursively parse
        has_fn_or_op = any(
            (isinstance(item, Name) and (item.name in self.PRIMITIVES or item.name in self.OPERATORS))
            or (isinstance(item, tuple) and item[0] == 'derived')
            for item in items
        )

        if has_fn_or_op:
            # Recursively parse the items using right-to-left evaluation
            return self._evaluate_items(items)
        else:
            # All values - create array
            return Array(elements=items)

    def _evaluate_items(self, items: list) -> Node:
        """Evaluate a list of items using right-to-left APL semantics."""
        if not items:
            return None
        if len(items) == 1:
            item = items[0]
            if isinstance(item, tuple) and item[0] == 'derived':
                _, fn, op = item
                return Name(name=f"{fn}{op}")
            return item

        result = items[-1]
        if isinstance(result, tuple) and result[0] == 'derived':
            _, fn, op = result
            result = Name(name=f"{fn}{op}")

        i = len(items) - 2

        while i >= 0:
            item = items[i]

            # Check for derived function
            if isinstance(item, tuple) and item[0] == 'derived':
                _, fn, op = item
                has_value_left = self._has_value_left(items, i)
                if i > 0 and has_value_left:
                    left = self._combine_items(items[:i])
                    result = DyadicOp(op=op, fn1=fn, fn2=fn, left=left, right=result)
                    return result
                else:
                    result = MonadicOp(op=op, fn=fn, right=result)
                    i -= 1
            elif isinstance(item, Name) and item.name in self.PRIMITIVES:
                fn = item.name
                has_value_left = self._has_value_left(items, i)
                if i > 0 and has_value_left:
                    # Dyadic - everything to the left is the left argument
                    left = self._combine_items(items[:i])
                    result = Dyad(fn=fn, left=left, right=result)
                    return result
                else:
                    # Monadic
                    result = Monad(fn=fn, right=result)
                    i -= 1
            elif isinstance(item, Name) and item.name in self.OPERATORS:
                # Standalone operator - skip
                i -= 1
            else:
                # Value - check if result is a function application
                if isinstance(result, (Monad, MonadicOp)):
                    if isinstance(result, Monad):
                        result = Dyad(fn=result.fn, left=item, right=result.right)
                    elif isinstance(result, MonadicOp):
                        result = DyadicOp(op=result.op, fn1=result.fn, fn2=result.fn,
                                        left=item, right=result.right)
                else:
                    if isinstance(result, Array):
                        result = Array(elements=[item] + result.elements)
                    else:
                        result = Array(elements=[item, result])
                i -= 1

        return result

    def _parse_primary(self) -> Node:
        """Parse a primary expression."""
        if self._at_end():
            return None

        token = self._current()

        # Number
        if token.type == TokenType.NUMBER:
            self._advance()
            return self._parse_number(token.value)

        # String
        if token.type == TokenType.STRING:
            self._advance()
            return Scalar(value=token.value)

        # Name (variable or primitive)
        if token.type == TokenType.NAME:
            self._advance()
            return Name(name=token.value)

        # Primitive function
        if token.type == TokenType.PRIMITIVE:
            self._advance()
            return Name(name=token.value)

        # Operator
        if token.type == TokenType.OPERATOR:
            self._advance()
            return Name(name=token.value)

        # Parenthesized expression
        if token.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            if self._check(TokenType.RPAREN):
                self._advance()
            return expr

        # Dfn
        if token.type == TokenType.LBRACE:
            return self._parse_dfn()

        # Indexed expression
        if token.type == TokenType.LBRACKET:
            return self._parse_index(None)

        return None

    def _parse_number(self, value: str) -> Scalar:
        """Parse a number string into a Scalar."""
        try:
            if 'j' in value.lower():
                return Scalar(value=complex(value))
            elif '.' in value or 'e' in value.lower():
                return Scalar(value=float(value))
            else:
                return Scalar(value=int(value))
        except ValueError:
            return Scalar(value=float(value))

    def _parse_dfn(self) -> Dfn:
        """Parse a direct function { body }."""
        self._advance()  # Skip {
        body = []

        while not self._at_end() and not self._check(TokenType.RBRACE):
            if self._check(TokenType.SEPARATOR):
                self._advance()
                continue

            # Check for guard
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)

        if self._check(TokenType.RBRACE):
            self._advance()

        return Dfn(body=body)

    def _parse_index(self, array: Node) -> Index:
        """Parse array indexing [indices]."""
        self._advance()  # Skip [
        indices = []

        while not self._at_end() and not self._check(TokenType.RBRACKET):
            if self._check(TokenType.SEPARATOR):
                self._advance()
                indices.append(None)  # Elided index
                continue

            idx = self._parse_expression()
            if idx:
                indices.append(idx)

        if self._check(TokenType.RBRACKET):
            self._advance()

        return Index(array=array, indices=indices)

    def _current(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, '')

    def _advance(self) -> Token:
        """Advance to next token and return current."""
        token = self._current()
        self.pos += 1
        return token

    def _check(self, type: TokenType) -> bool:
        """Check if current token is of given type."""
        return self._current().type == type

    def _check_ahead(self, type: TokenType, offset: int) -> bool:
        """Check token at offset ahead."""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos].type == type
        return False

    def _at_end(self) -> bool:
        """Check if at end of tokens."""
        return self._current().type == TokenType.EOF


def parse(source: str) -> list[Node]:
    """Parse APL source code into AST."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()
