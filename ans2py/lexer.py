"""APL Lexer - Tokenize APL source code."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


class TokenType(Enum):
    """APL token types."""
    NUMBER = auto()
    STRING = auto()
    NAME = auto()
    PRIMITIVE = auto()
    OPERATOR = auto()
    ASSIGN = auto()
    BRANCH = auto()
    SEPARATOR = auto()
    COMMENT = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    NEWLINE = auto()
    COLON = auto()
    EOF = auto()


@dataclass
class Token:
    """A single APL token."""
    type: TokenType
    value: str
    line: int = 1
    column: int = 1

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r})"


# APL primitive functions
PRIMITIVES = set('+-×÷*⍟!○|⌈⌊⊥⊤⊣⊢<≤=≥>≠≡≢∧∨⍲⍱~⍳⍴⍪,⌽⊖⍉↑↓⊂⊃∊⍷⍸⌷⍋⍒?⍎⍕⌹∪∩⍺⍵∇')

# APL operators
OPERATORS = set('/\\¨⍨⍣⍤⍥∘.@⌸⌺⍠')

# High minus for negative numbers
HIGH_MINUS = '¯'


class Lexer:
    """Tokenize APL source code."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source."""
        tokens = []
        for token in self._generate_tokens():
            tokens.append(token)
        return tokens

    def _generate_tokens(self) -> Iterator[Token]:
        """Generate tokens from source."""
        while self.pos < len(self.source):
            char = self.source[self.pos]

            if char in ' \t':
                self._advance()
                continue

            if char == '\n':
                yield self._make_token(TokenType.NEWLINE, '\n')
                self._advance()
                self.line += 1
                self.column = 1
                continue

            if char == '\r':
                self._advance()
                continue

            if char == '⍝':
                start = self.pos
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self._advance()
                yield self._make_token(TokenType.COMMENT, self.source[start:self.pos])
                continue

            if char == "'":
                yield self._read_string()
                continue

            if char.isdigit() or char == HIGH_MINUS or (char == '.' and self._peek_digit()):
                yield self._read_number()
                continue

            if char == '←':
                yield self._make_token(TokenType.ASSIGN, char)
                self._advance()
                continue

            if char == '→':
                yield self._make_token(TokenType.BRANCH, char)
                self._advance()
                continue

            if char == '⋄':
                yield self._make_token(TokenType.SEPARATOR, char)
                self._advance()
                continue

            if char == ':':
                yield self._make_token(TokenType.COLON, char)
                self._advance()
                continue

            if char == '(':
                yield self._make_token(TokenType.LPAREN, char)
                self._advance()
                continue
            if char == ')':
                yield self._make_token(TokenType.RPAREN, char)
                self._advance()
                continue
            if char == '[':
                yield self._make_token(TokenType.LBRACKET, char)
                self._advance()
                continue
            if char == ']':
                yield self._make_token(TokenType.RBRACKET, char)
                self._advance()
                continue
            if char == '{':
                yield self._make_token(TokenType.LBRACE, char)
                self._advance()
                continue
            if char == '}':
                yield self._make_token(TokenType.RBRACE, char)
                self._advance()
                continue

            if char in OPERATORS:
                if char == '∘' and self._peek() == '.':
                    self._advance()
                    self._advance()
                    yield self._make_token(TokenType.OPERATOR, '∘.')
                    continue
                yield self._make_token(TokenType.OPERATOR, char)
                self._advance()
                continue

            if char in PRIMITIVES:
                yield self._make_token(TokenType.PRIMITIVE, char)
                self._advance()
                continue

            if char.isalpha() or char == '_' or char == '∆' or char == '⍙':
                yield self._read_name()
                continue

            if char == '⎕':
                yield self._read_quad_name()
                continue

            self._advance()

        yield self._make_token(TokenType.EOF, '')

    def _advance(self):
        self.pos += 1
        self.column += 1

    def _peek(self) -> str:
        if self.pos + 1 < len(self.source):
            return self.source[self.pos + 1]
        return ''

    def _peek_digit(self) -> bool:
        return self._peek().isdigit()

    def _make_token(self, type: TokenType, value: str) -> Token:
        return Token(type, value, self.line, self.column)

    def _read_string(self) -> Token:
        start_col = self.column
        self._advance()
        chars = []

        while self.pos < len(self.source):
            char = self.source[self.pos]
            if char == "'":
                if self._peek() == "'":
                    chars.append("'")
                    self._advance()
                    self._advance()
                else:
                    self._advance()
                    break
            else:
                chars.append(char)
                self._advance()

        return Token(TokenType.STRING, ''.join(chars), self.line, start_col)

    def _read_number(self) -> Token:
        start_col = self.column
        chars = []

        if self.source[self.pos] == HIGH_MINUS:
            chars.append('-')
            self._advance()

        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            chars.append(self.source[self.pos])
            self._advance()

        if self.pos < len(self.source) and self.source[self.pos] == '.':
            chars.append('.')
            self._advance()
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                chars.append(self.source[self.pos])
                self._advance()

        if self.pos < len(self.source) and self.source[self.pos] in 'Ee':
            chars.append('E')
            self._advance()
            if self.pos < len(self.source) and self.source[self.pos] in (HIGH_MINUS, '-'):
                chars.append('-')
                self._advance()
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                chars.append(self.source[self.pos])
                self._advance()

        if self.pos < len(self.source) and self.source[self.pos] in 'Jj':
            chars.append('j')
            self._advance()
            if self.pos < len(self.source) and self.source[self.pos] == HIGH_MINUS:
                chars.append('-')
                self._advance()
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                chars.append(self.source[self.pos])
                self._advance()
            if self.pos < len(self.source) and self.source[self.pos] == '.':
                chars.append('.')
                self._advance()
                while self.pos < len(self.source) and self.source[self.pos].isdigit():
                    chars.append(self.source[self.pos])
                    self._advance()

        return Token(TokenType.NUMBER, ''.join(chars), self.line, start_col)

    def _read_name(self) -> Token:
        start_col = self.column
        chars = []

        while self.pos < len(self.source):
            char = self.source[self.pos]
            if char.isalnum() or char in '_∆⍙':
                chars.append(char)
                self._advance()
            else:
                break

        return Token(TokenType.NAME, ''.join(chars), self.line, start_col)

    def _read_quad_name(self) -> Token:
        start_col = self.column
        chars = ['⎕']
        self._advance()

        while self.pos < len(self.source):
            char = self.source[self.pos]
            if char.isalnum() or char == '_':
                chars.append(char)
                self._advance()
            else:
                break

        return Token(TokenType.NAME, ''.join(chars), self.line, start_col)


def tokenize(source: str) -> list[Token]:
    """Tokenize APL source code."""
    return Lexer(source).tokenize()
