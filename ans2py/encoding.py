"""APL2 character encoding - maps IBM APL2 bytes to Unicode."""

# IBM APL2 byte to Unicode mapping
# Based on IBM APL2 character set documentation
APL2_TO_UNICODE = {
    # Control and special
    0x00: '\x00',  # Null
    0x0A: '\n',    # Line feed
    0x0D: '\r',    # Carriage return

    # Space and basic punctuation
    0x20: ' ',     # Space
    0x21: '!',     # Exclamation / Factorial
    0x22: '"',     # Quote
    0x23: '#',     # Hash
    0x24: '$',     # Dollar
    0x25: '%',     # Percent
    0x26: '&',     # Ampersand
    0x27: "'",     # Apostrophe
    0x28: '(',     # Left paren
    0x29: ')',     # Right paren
    0x2A: '*',     # Asterisk / Power
    0x2B: '+',     # Plus
    0x2C: ',',     # Comma / Catenate
    0x2D: '-',     # Minus
    0x2E: '.',     # Dot / Inner product
    0x2F: '/',     # Slash / Reduce

    # Digits
    0x30: '0', 0x31: '1', 0x32: '2', 0x33: '3', 0x34: '4',
    0x35: '5', 0x36: '6', 0x37: '7', 0x38: '8', 0x39: '9',

    # Punctuation
    0x3A: ':',     # Colon
    0x3B: ';',     # Semicolon
    0x3C: '<',     # Less than
    0x3D: '=',     # Equal
    0x3E: '>',     # Greater than
    0x3F: '?',     # Question / Roll
    0x40: '@',     # At

    # Uppercase letters
    0x41: 'A', 0x42: 'B', 0x43: 'C', 0x44: 'D', 0x45: 'E',
    0x46: 'F', 0x47: 'G', 0x48: 'H', 0x49: 'I', 0x4A: 'J',
    0x4B: 'K', 0x4C: 'L', 0x4D: 'M', 0x4E: 'N', 0x4F: 'O',
    0x50: 'P', 0x51: 'Q', 0x52: 'R', 0x53: 'S', 0x54: 'T',
    0x55: 'U', 0x56: 'V', 0x57: 'W', 0x58: 'X', 0x59: 'Y',
    0x5A: 'Z',

    # Brackets and special
    0x5B: '[',     # Left bracket
    0x5C: '\\',    # Backslash / Scan
    0x5D: ']',     # Right bracket
    0x5E: '^',     # Caret
    0x5F: '_',     # Underscore
    0x60: '`',     # Backtick

    # Lowercase letters
    0x61: 'a', 0x62: 'b', 0x63: 'c', 0x64: 'd', 0x65: 'e',
    0x66: 'f', 0x67: 'g', 0x68: 'h', 0x69: 'i', 0x6A: 'j',
    0x6B: 'k', 0x6C: 'l', 0x6D: 'm', 0x6E: 'n', 0x6F: 'o',
    0x70: 'p', 0x71: 'q', 0x72: 'r', 0x73: 's', 0x74: 't',
    0x75: 'u', 0x76: 'v', 0x77: 'w', 0x78: 'x', 0x79: 'y',
    0x7A: 'z',

    # Braces
    0x7B: '{',     # Left brace
    0x7C: '|',     # Bar / Absolute
    0x7D: '}',     # Right brace
    0x7E: '~',     # Tilde / Not

    # APL special characters (high bytes)
    0x80: '¨',     # Diaeresis / Each
    0x81: '¯',     # High minus (negative)
    0x82: '×',     # Multiply sign
    0x83: '÷',     # Divide sign
    0x84: '←',     # Left arrow / Assignment
    0x85: '→',     # Right arrow / Branch
    0x86: '↑',     # Up arrow / Take
    0x87: '↓',     # Down arrow / Drop
    0x88: '∆',     # Delta (user identifier)
    0x89: '∇',     # Del (function definition)
    0x8A: '∘',     # Jot / Compose
    0x8B: '∧',     # And
    0x8C: '∨',     # Or
    0x8D: '∩',     # Intersection
    0x8E: '∪',     # Union
    0x8F: '⊂',     # Enclose

    0x90: '⊃',     # Disclose
    0x91: '⊤',     # Encode
    0x92: '⊥',     # Decode
    0x93: '⌈',     # Ceiling / Max
    0x94: '⌊',     # Floor / Min
    0x95: '⌶',     # I-beam
    0x96: '⌷',     # Index
    0x97: '⌹',     # Matrix inverse/divide
    0x98: '⌽',     # Reverse/Rotate
    0x99: '⌿',     # Reduce first
    0x9A: '⍀',     # Scan first
    0x9B: '⍉',     # Transpose
    0x9C: '⍋',     # Grade up
    0x9D: '⍎',     # Execute
    0x9E: '⍒',     # Grade down
    0x9F: '⍕',     # Format

    0xA0: '⍝',     # Comment
    0xA1: '⍞',     # Quote-quad
    0xA2: '⍟',     # Log
    0xA3: '⍣',     # Power operator
    0xA4: '⍤',     # Rank
    0xA5: '⍥',     # Over
    0xA6: '⍨',     # Commute
    0xA7: '⍪',     # Table/Catenate first
    0xA8: '⍫',     # Del tilde
    0xA9: '⍬',     # Zilde (empty numeric)
    0xAA: '⍱',     # Nor
    0xAB: '⍲',     # Nand
    0xAC: '⍳',     # Iota
    0xAD: '⍴',     # Rho / Shape
    0xAE: '⍵',     # Omega (right arg)
    0xAF: '⍶',     # Alpha underbar

    0xB0: '⍷',     # Find
    0xB1: '⍸',     # Where
    0xB2: '⍹',     # Omega underbar
    0xB3: '⍺',     # Alpha (left arg)
    0xB4: '⎕',     # Quad (system)
    0xB5: '○',     # Circle / Pi times / Trig
    0xB6: '≠',     # Not equal
    0xB7: '≡',     # Match / Depth
    0xB8: '≢',     # Not match / Tally
    0xB9: '≤',     # Less or equal
    0xBA: '≥',     # Greater or equal
    0xBB: '⊣',     # Left
    0xBC: '⊢',     # Right
    0xBD: '←',     # Assignment (alternate)
    0xBE: '⋄',     # Statement separator
    0xBF: '⍺',     # Alpha (alternate)

    # More APL characters
    0xC0: '∊',     # Epsilon / Membership
    0xC1: '⍷',     # Find (alternate)
    0xC2: '⌻',     # Quad divide
    0xC3: '⌼',     # Circle jot
    0xC4: '⌾',     # Circle star
    0xC5: '⍁',     # Quad slash
    0xC6: '⍂',     # Quad backslash
    0xC7: '⍃',     # Quad less
    0xC8: '⍄',     # Quad greater
    0xC9: '⍅',     # Left vane
    0xCA: '⍆',     # Right vane
    0xCB: '⍇',     # Quad left arrow
    0xCC: '⍈',     # Quad right arrow
    0xCD: '⍊',     # Tack up
    0xCE: '⍌',     # Circle stile
    0xCF: '⍍',     # Circle backslash

    0xD0: '⍏',     # Quad up
    0xD1: '⍐',     # Quad down
    0xD2: '⍑',     # Del stile
    0xD3: '⍓',     # Circle minus
    0xD4: '⍔',     # Circle plus
    0xD5: '⍖',     # Up arrow stile
    0xD6: '⍗',     # Down arrow stile
    0xD7: '⍘',     # Quote quad
    0xD8: '⍙',     # Delta underbar
    0xD9: '⍚',     # Diamond underbar
    0xDA: '⍛',     # Jot underbar
    0xDB: '⍜',     # Circle underbar
    0xDC: '⍟',     # Star diaeresis
    0xDD: '⍠',     # Quad colon
    0xDE: '⍡',     # Tack diaeresis
    0xDF: '⍢',     # Del diaeresis

    # Extended APL
    0xE0: '⍤',     # Jot diaeresis / Rank
    0xE1: '⍥',     # Circle diaeresis
    0xE2: '⍦',     # Star diaeresis 2
    0xE3: '⍧',     # Quad equal
    0xE4: '⍝',     # Lamp / Comment (alternate)
    0xE5: '⍩',     # Comma bar
    0xE6: '⍪',     # Comma bar (alternate)
    0xE7: '∈',     # Element of
    0xE8: '⍨',     # Tilde diaeresis
    0xE9: '⍠',     # Variant
    0xEA: '⍢',     # Power operator
    0xEB: '∘',     # Circle / Compose
    0xEC: '×',     # Times (alternate)
    0xED: '⍺',     # Alpha (alternate 2)
    0xEE: '⍵',     # Omega (alternate)
    0xEF: '⍶',     # Alpha underbar (alternate)

    0xF0: '⍹',     # Omega underbar (alternate)
    0xF1: '⍳',     # Iota (alternate)
    0xF2: '⍴',     # Rho (alternate)
    0xF3: '⍬',     # Zilde (alternate)
    0xF4: '⎕',     # Quad (alternate)
    0xF5: '×',     # Times (alternate 2)
    0xF6: '÷',     # Divide (alternate)
    0xF7: '⌹',     # Domino (alternate)
    0xF8: '⊖',     # Circle stile / Rotate first
    0xF9: '⍀',     # Backslash bar (alternate)
    0xFA: '⌿',     # Slash bar (alternate)
    0xFB: '⍕',     # Format (alternate)
    0xFC: '⍎',     # Execute (alternate)
    0xFD: '⍸',     # Where (alternate)
    0xFE: '⍷',     # Find (alternate)
    0xFF: '⍨',     # Selfie (alternate)
}

# Create reverse mapping
UNICODE_TO_APL2 = {v: k for k, v in APL2_TO_UNICODE.items()}


def decode_apl_bytes(data: bytes) -> str:
    """Decode IBM APL2 encoded bytes to Unicode string."""
    result = []
    for byte in data:
        if byte in APL2_TO_UNICODE:
            result.append(APL2_TO_UNICODE[byte])
        elif 0x20 <= byte < 0x7F:
            # Printable ASCII fallback
            result.append(chr(byte))
        else:
            # Unknown byte - use replacement character
            result.append(f'\\x{byte:02x}')
    return ''.join(result)


def encode_to_apl_bytes(text: str) -> bytes:
    """Encode Unicode APL text to IBM APL2 bytes."""
    result = []
    for char in text:
        if char in UNICODE_TO_APL2:
            result.append(UNICODE_TO_APL2[char])
        elif ord(char) < 128:
            result.append(ord(char))
        else:
            # Unknown character
            result.append(0x3F)  # Question mark
    return bytes(result)
