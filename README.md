# ans2py

Convert IBM APL2 `.ANS` workspace files and APL expressions to Python/NumPy code.

## Features

- **Parse ANS files** - Extract function names, signatures, and documentation from IBM APL2 binary workspaces
- **Convert APL to Python** - Transform APL expressions into readable, executable Python/NumPy code
- **ASCII output** - Generated Python is fully ASCII with no APL glyphs in executable code
- **NumPy runtime** - Includes a runtime library implementing APL primitives using NumPy

## Installation

```bash
pip install -e .
```

## Usage

### Convert APL expressions to Python

```bash
ans2py convert --apl "+/ ⍳ 5"
```

Output:
```python
import numpy as np
from ans2py.runtime import apl_iota, apl_reduce

apl_reduce('add', apl_iota(5))
```

### More examples

```bash
# Reshape and reduce
ans2py convert --apl "2 3 ⍴ ⍳ 6"
# → apl_reshape(np.array([2, 3]), apl_iota(6))

# Max reduce
ans2py convert --apl "⌈/ 1 5 3 2 4"
# → apl_reduce('max', np.array([1, 5, 3, 2, 4]))

# Right-to-left evaluation
ans2py convert --apl "3 × 2 + 1"
# → np.multiply(3, np.add(2, 1))  # = 9
```

### Convert ANS workspace to Python stubs

```bash
ans2py convert MATHFNS.ans -o mathfns.py
```

Generates Python function stubs with docstrings:
```python
def domino(alpha, omega):
    """
    APL: DOMINO
    Signature: Z←L DOMINO R
    """
    raise NotImplementedError("DOMINO not yet implemented")

def fft(omega):
    """
    APL: FFT
    Signature: Z←FFT R
    """
    raise NotImplementedError("FFT not yet implemented")
```

### List functions in a workspace

```bash
ans2py extract MATHFNS.ans --list
```

### Get workspace info

```bash
ans2py info MATHFNS.ans
```

Output:
```
File: MATHFNS.ans
Size: 200,128 bytes
Format: ANS 1.00
Workspace: APL2 WS
Functions/Variables: 27
```

## Supported APL Primitives

| APL | Python | Description |
|-----|--------|-------------|
| `⍳ N` | `apl_iota(N)` | Index generator (1 to N) |
| `⍴` | `apl_reshape` | Reshape array |
| `+/ -/ ×/ ÷/` | `apl_reduce` | Reduce with operation |
| `⌈/ ⌊/` | `apl_reduce('max'/min')` | Max/min reduce |
| `+\ ×\` | `np.cumsum/cumprod` | Scan (cumulative) |
| `+ - × ÷` | `np.add/subtract/multiply/divide` | Arithmetic |
| `⌈ ⌊` | `np.maximum/minimum` | Max/min |
| `←` | `=` | Assignment |

## Project Structure

```
ans2py/
├── ans2py/
│   ├── __init__.py
│   ├── ans_parser.py   # ANS binary file parser
│   ├── lexer.py        # APL tokenizer
│   ├── parser.py       # APL AST builder (right-to-left)
│   ├── generator.py    # Python code generator
│   ├── runtime.py      # NumPy-based APL runtime
│   ├── encoding.py     # APL character encoding
│   └── cli.py          # Command-line interface
├── tests/
│   ├── fixtures/
│   │   └── MATHFNS.ans # Sample workspace
│   └── test_ans_parser.py
├── pyproject.toml
└── README.md
```

## Tested Workspaces

Successfully converts all 16 workspaces from IBM APL2:

| Workspace | Functions | Description |
|-----------|-----------|-------------|
| MATHFNS.ans | 12 | Math (FFT, eigenvalues, matrix ops) |
| GRAPHPAK.ans | 139 | Graphics package |
| GUITOOLS.ans | 92 | GUI toolkit |
| APL2LM.ans | 96 | Library manager |
| SQL.ans | 62 | Database interface |
| FILE.ans | 28 | File I/O |
| ... | ... | ... |

## License

MIT
