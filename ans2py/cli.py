"""Command-line interface for ans2py."""

import argparse
import sys
from pathlib import Path

from .ans_parser import ANSParser, parse_ans_file
from .generator import generate_python, convert_apl_to_python


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog='ans2py',
        description='Convert IBM APL2 .ANS workspace files to Python'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract APL source code from .ANS file'
    )
    extract_parser.add_argument('input', help='Input .ANS file')
    extract_parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    extract_parser.add_argument('--list', action='store_true', help='List function names only')
    extract_parser.add_argument('--function', '-f', help='Extract specific function')
    extract_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show workspace information'
    )
    info_parser.add_argument('input', help='Input .ANS file')

    # Convert command
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert APL to Python'
    )
    convert_parser.add_argument('input', nargs='?', help='Input .ANS or .apl file')
    convert_parser.add_argument('-o', '--output', help='Output Python file')
    convert_parser.add_argument('--apl', '-a', help='APL code string to convert')
    convert_parser.add_argument('--function', '-f', help='Wrap output in function with this name')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == 'extract':
            cmd_extract(args)
        elif args.command == 'info':
            cmd_info(args)
        elif args.command == 'convert':
            cmd_convert(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_extract(args):
    """Handle extract command."""
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    parser = ANSParser(input_path)
    workspace = parser.parse()

    if args.list:
        # Just list function names
        functions = parser.list_functions()
        output = '\n'.join(functions)
    elif args.function:
        # Extract specific function
        if args.function not in workspace.functions:
            # Try case-insensitive match
            matches = [f for f in workspace.raw_symbols
                      if f.upper() == args.function.upper()]
            if matches:
                output = parser.get_function_source(matches[0])
            else:
                print(f"Function '{args.function}' not found.", file=sys.stderr)
                print(f"Available functions: {', '.join(workspace.raw_symbols[:20])}", file=sys.stderr)
                sys.exit(1)
        else:
            output = parser.get_function_source(args.function)
    else:
        # Extract all content
        ws_name = ''.join(c for c in workspace.name if c.isprintable()).strip()
        lines = []
        lines.append(f"⍝ Workspace: {ws_name}")
        lines.append(f"⍝ Version: ANS {workspace.version}")
        lines.append(f"⍝ Symbols: {len(workspace.raw_symbols)}")
        lines.append("")

        # Get function signatures
        sigs = parser.extract_function_signatures()

        # List functions with signatures
        func_names = [n for n in workspace.raw_symbols if len(n) > 2 and n.isupper()]
        if func_names:
            lines.append("⍝ " + "=" * 60)
            lines.append("⍝ FUNCTION SIGNATURES")
            lines.append("⍝ " + "=" * 60)
            lines.append("")
            for name in sorted(func_names):
                sig = sigs.get(name, f"∇ {name}")
                lines.append(f"∇ {sig}")
                lines.append("  ⍝ [function body]")
                lines.append("∇")
                lines.append("")

        # Extract documentation
        lines.append("⍝ " + "=" * 60)
        lines.append("⍝ DOCUMENTATION")
        lines.append("⍝ " + "=" * 60)
        lines.append("")
        lines.append(parser.extract_all_apl_text())

        output = '\n'.join(lines)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output, encoding='utf-8')
        print(f"Written to {output_path}")
    else:
        print(output)


def cmd_info(args):
    """Handle info command."""
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    parser = ANSParser(input_path)
    workspace = parser.parse()

    # Clean workspace name
    ws_name = ''.join(c for c in workspace.name if c.isprintable()).strip()

    print(f"File: {input_path}")
    print(f"Size: {input_path.stat().st_size:,} bytes")
    print(f"Format: ANS {workspace.version}")
    print(f"Workspace: {ws_name}")
    print(f"Functions/Variables: {len(workspace.raw_symbols)}")
    print()
    print("Symbols:")
    for i, name in enumerate(sorted(workspace.raw_symbols)):
        print(f"  {name}")
        if i >= 49:
            remaining = len(workspace.raw_symbols) - 50
            if remaining > 0:
                print(f"  ... and {remaining} more")
            break


import keyword

# Python keywords and builtins that need to be renamed
PYTHON_RESERVED = set(keyword.kwlist) | {
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray',
    'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
    'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec',
    'filter', 'float', 'format', 'frozenset', 'getattr', 'globals',
    'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
    'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow',
    'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
    'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super',
    'tuple', 'type', 'vars', 'zip'
}


def sanitize_python_name(name: str) -> str:
    """Convert APL function name to valid Python identifier."""
    py_name = name.lower().replace('_', '_')
    if py_name in PYTHON_RESERVED:
        return f'apl_{py_name}'
    return py_name


def cmd_convert(args):
    """Handle convert command."""
    # Get APL source code
    if args.apl:
        # Direct APL code from command line
        apl_source = args.apl
        if args.function:
            output = convert_apl_to_python(apl_source, args.function)
        else:
            output = generate_python(apl_source)
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        if input_path.suffix.lower() == '.ans':
            # Extract from ANS file
            parser = ANSParser(input_path)
            workspace = parser.parse()

            # Get function signatures and documentation
            sigs = parser.extract_function_signatures()

            # Generate Python stubs based on signatures
            # Clean workspace name (remove null bytes and control characters)
            ws_name = ''.join(c for c in workspace.name if c.isprintable()).strip()

            lines = []
            lines.append('"""')
            lines.append(f'Converted from: {input_path.name}')
            lines.append(f'Workspace: {ws_name}')
            lines.append('"""')
            lines.append('')
            lines.append('import numpy as np')
            lines.append('from ans2py.runtime import *')
            lines.append('')

            func_names = [n for n in workspace.raw_symbols if len(n) > 2 and n.isupper()]
            for name in sorted(func_names):
                py_name = sanitize_python_name(name)
                sig = sigs.get(name, '')
                if 'L' in sig:
                    # Dyadic function
                    lines.append(f'def {py_name}(alpha, omega):')
                else:
                    # Monadic function
                    lines.append(f'def {py_name}(omega):')
                lines.append(f'    """')
                lines.append(f'    APL: {name}')
                if sig:
                    lines.append(f'    Signature: {sig}')
                lines.append(f'    """')
                lines.append(f'    # TODO: Implement {name}')
                lines.append(f'    raise NotImplementedError("{name} not yet implemented")')
                lines.append('')

            output = '\n'.join(lines)
        else:
            # Read APL source file
            apl_source = input_path.read_text(encoding='utf-8')

            # Convert APL to Python
            if args.function:
                output = convert_apl_to_python(apl_source, args.function)
            else:
                output = generate_python(apl_source)
    else:
        print("Error: Either input file or --apl argument required", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output, encoding='utf-8')
        print(f"Written to {output_path}")
    else:
        print(output)


if __name__ == '__main__':
    main()
