"""Parser for IBM APL2 .ANS workspace files."""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from .encoding import decode_apl_bytes


@dataclass
class APLFunction:
    """Represents an APL function extracted from workspace."""
    name: str
    source: str = ""
    header: str = ""
    comments: list[str] = field(default_factory=list)
    line_count: int = 0


@dataclass
class APLVariable:
    """Represents an APL variable."""
    name: str
    value: any = None
    shape: tuple = ()
    type_code: int = 0


@dataclass
class ANSWorkspace:
    """Represents a parsed ANS workspace."""
    version: str = ""
    name: str = ""
    functions: dict[str, APLFunction] = field(default_factory=dict)
    variables: dict[str, APLVariable] = field(default_factory=dict)
    description: str = ""
    raw_symbols: list[str] = field(default_factory=list)


class ANSParser:
    """Parser for IBM APL2 .ANS workspace files."""

    HEADER_MAGIC = b'ANS 1.00'
    WS_MARKER = b'APL2 WS'

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.data: bytes = b''
        self.workspace = ANSWorkspace()

    def parse(self) -> ANSWorkspace:
        """Parse the ANS file and return workspace contents."""
        with open(self.filepath, 'rb') as f:
            self.data = f.read()

        self._validate_header()
        self._parse_workspace_info()
        self._extract_symbols()
        self._extract_functions()

        return self.workspace

    def _validate_header(self) -> None:
        """Validate ANS file header."""
        if not self.data.startswith(self.HEADER_MAGIC):
            raise ValueError(f"Invalid ANS file: expected header '{self.HEADER_MAGIC.decode()}'")

        self.workspace.version = self.data[4:8].decode('ascii').strip()

        # Find APL2 WS marker
        ws_pos = self.data.find(self.WS_MARKER)
        if ws_pos == -1:
            raise ValueError("Invalid ANS file: APL2 WS marker not found")

    def _parse_workspace_info(self) -> None:
        """Parse workspace metadata."""
        # Find workspace name (after APL2 WS marker)
        ws_pos = self.data.find(self.WS_MARKER)
        if ws_pos != -1:
            # Workspace name follows the marker
            name_area = self.data[ws_pos:ws_pos + 16]
            self.workspace.name = name_area.decode('ascii', errors='ignore').strip()

    def _extract_symbols(self) -> None:
        """Extract symbol names from the symbol table."""
        symbols = []

        # Look for function/variable names in the binary
        # Names typically appear as sequences of uppercase letters
        # preceded by a length byte

        i = 0x80  # Skip header area
        while i < len(self.data) - 2:
            # Look for name pattern: 0x02 + length + name bytes
            if self.data[i] == 0x02:
                length = self.data[i + 1]
                if 1 <= length <= 32:
                    name_bytes = self.data[i + 2:i + 2 + length]
                    # Check if it looks like a valid name
                    if all(32 <= b < 127 for b in name_bytes):
                        try:
                            name = name_bytes.decode('ascii').strip('\x00')
                            if name and name.replace('_', '').isalnum():
                                if name not in symbols:
                                    symbols.append(name)
                        except UnicodeDecodeError:
                            pass
            i += 1

        self.workspace.raw_symbols = symbols

    def _extract_functions(self) -> None:
        """Extract function definitions from the workspace."""
        # Look for function header patterns like "Z←L FUNCNAME R"
        # or "Z← FUNCNAME R" (monadic)

        # First, find text blocks that contain function definitions
        text_blocks = self._find_text_blocks()

        for name in self.workspace.raw_symbols:
            func = APLFunction(name=name)

            # Look for function header containing the name
            for block in text_blocks:
                if name in block:
                    # Check if this looks like a function header
                    if '←' in block or 'bd' in block.lower():
                        func.header = block
                        break

            # Look for comments mentioning this function
            for block in text_blocks:
                if name in block and len(block) > 20:
                    func.comments.append(block)

            if func.header or func.comments:
                self.workspace.functions[name] = func

    def _find_text_blocks(self) -> list[str]:
        """Find readable text blocks in the binary data."""
        blocks = []
        current_block = []
        min_length = 10

        # APL byte to Unicode mapping for common symbols
        apl_chars = {
            0xBD: '←', 0x84: '←',  # Assignment
            0xAC: '⍳',  # Iota
            0xAD: '⍴',  # Rho
            0xB5: '○',  # Circle
            0x93: '⌈',  # Ceiling
            0x94: '⌊',  # Floor
            0x82: '×',  # Times
            0x83: '÷',  # Divide
            0xAE: '⍵',  # Omega
            0xED: '⍵',  # Omega alternate
            0xB3: '⍺',  # Alpha
            0xBF: '⍺',  # Alpha alternate
            0x89: '∇',  # Del
            0xA0: '⍝',  # Comment
            0xE4: '⍝',  # Comment alternate
            0x8E: '⍝',  # Comment alternate 2
            0x86: '↑',  # Take
            0x87: '↓',  # Drop
            0x8B: '∧',  # And
            0x8C: '∨',  # Or
            0x98: '⌽',  # Rotate
            0x9B: '⍉',  # Transpose
            0x97: '⌹',  # Domino
            0xF7: '⌹',  # Domino alternate
            0xB6: '≠',  # Not equal
            0xB9: '≤',  # Less or equal
            0xBA: '≥',  # Greater or equal
            0xC0: '∊',  # Membership
            0xF5: '×',  # Times alternate
            0xF6: '÷',  # Divide alternate
            0xF8: '⊖',  # Rotate first
            0xEC: '×',  # Times alternate 2
            0xF3: '⌷',  # Index
            0xA9: '⍬',  # Zilde
            0x92: '⊥',  # Decode
            0x91: '⊤',  # Encode
        }

        i = 0
        while i < len(self.data):
            byte = self.data[i]

            # Check for printable ASCII
            if 32 <= byte < 127:
                current_block.append(chr(byte))
            elif byte in (0x0A, 0x0D):  # Newlines
                if current_block:
                    current_block.append(' ')
            elif byte in apl_chars:
                current_block.append(apl_chars[byte])
            else:
                # End of text block
                if len(current_block) >= min_length:
                    text = ''.join(current_block).strip()
                    if text and not text.isspace():
                        blocks.append(text)
                current_block = []

            i += 1

        # Don't forget the last block
        if len(current_block) >= min_length:
            text = ''.join(current_block).strip()
            if text and not text.isspace():
                blocks.append(text)

        return blocks

    def extract_all_apl_text(self) -> str:
        """Extract all readable APL text from the workspace."""
        blocks = self._find_text_blocks()

        # Filter out very short blocks and duplicates
        seen = set()
        result = []

        for block in blocks:
            # Clean up the block
            block = block.strip()
            if len(block) < 15:
                continue
            if block in seen:
                continue
            seen.add(block)

            # Check if it contains function-like content
            if '←' in block or '⍳' in block or '⍴' in block or 'DOMINO' in block or 'EIGEN' in block:
                result.append(block)
            elif any(name in block for name in self.workspace.raw_symbols if len(name) > 2):
                result.append(block)
            elif len(block) > 50:  # Longer text blocks are likely documentation
                result.append(block)

        return '\n\n'.join(result)

    def extract_function_signatures(self) -> dict[str, str]:
        """Extract function signatures like 'Z←L DOMINO R'."""
        import re

        signatures = {}  # name -> {'monadic': sig, 'dyadic': sig}

        # APL byte mapping
        apl_chars = {
            0xBD: '←', 0x84: '←', 0xAC: '⍳', 0xAD: '⍴',
            0x93: '⌈', 0x94: '⌊', 0x82: '×', 0x83: '÷',
            0xAE: '⍵', 0xB3: '⍺', 0x89: '∇', 0xA0: '⍝',
            0x97: '⌹', 0xB6: '≠', 0xF5: '×', 0xF6: '÷',
        }

        # Decode the description region (0x1d000 to 0x21000)
        result = []
        for b in self.data[0x1d000:0x21000]:
            if 32 <= b < 127:
                result.append(chr(b))
            elif b in apl_chars:
                result.append(apl_chars[b])
            elif b in (0x0A, 0x0D):
                result.append('\n')
            elif b == 0x00:
                pass
            else:
                result.append(' ')
        text = ''.join(result)

        # Find all function signature patterns for known function names
        for name in self.workspace.raw_symbols:
            if len(name) < 3 or not name.isupper():
                continue

            # Look for dyadic form: Z←L NAME R
            dyadic_pattern = rf'Z←L\s+{re.escape(name)}\s+R'
            dyadic_match = re.search(dyadic_pattern, text)

            # Look for monadic form: Z← NAME R or Z←NAME R
            monadic_pattern = rf'Z←\s*{re.escape(name)}\s*R?'
            monadic_match = re.search(monadic_pattern, text)

            if dyadic_match:
                # Normalize the dyadic signature
                sig = dyadic_match.group(0)
                sig = re.sub(r'\s+', ' ', sig)
                signatures[name] = sig
            elif monadic_match:
                # Normalize the monadic signature
                sig = monadic_match.group(0)
                sig = re.sub(r'\s+', ' ', sig)
                if not sig.endswith('R') and f'{name} R' in text:
                    sig = f"Z←{name} R"
                signatures[name] = sig

        return signatures

    def get_function_info(self, name: str) -> dict:
        """Get detailed info about a function."""
        if not hasattr(self, '_signatures'):
            self._signatures = self.extract_function_signatures()

        info = {
            'name': name,
            'signature': self._signatures.get(name, f'∇ {name}'),
            'comments': [],
            'documentation': '',
        }

        # Find documentation for this function
        blocks = self._find_text_blocks()
        for block in blocks:
            if name in block:
                if len(block) > 100:
                    info['documentation'] = block
                elif '⍝' in block or block.startswith(name):
                    info['comments'].append(block)

        return info

    def list_functions(self) -> list[str]:
        """Return list of function names in the workspace."""
        if not self.workspace.raw_symbols:
            self.parse()
        return sorted(self.workspace.raw_symbols)

    def get_function_source(self, name: str) -> str:
        """Get the source code for a specific function."""
        if name not in self.workspace.functions:
            return f"Function '{name}' not found"

        func = self.workspace.functions[name]
        lines = []

        if func.header:
            lines.append(f"∇ {func.header}")

        if func.comments:
            for comment in func.comments[:5]:  # Limit comments
                if len(comment) < 200:
                    lines.append(f"  ⍝ {comment}")

        lines.append("∇")

        return '\n'.join(lines)


def parse_ans_file(filepath: str | Path) -> ANSWorkspace:
    """Parse an ANS file and return the workspace."""
    parser = ANSParser(filepath)
    return parser.parse()


def list_workspace_functions(filepath: str | Path) -> list[str]:
    """List all function names in an ANS workspace."""
    parser = ANSParser(filepath)
    return parser.list_functions()
