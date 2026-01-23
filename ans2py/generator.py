"""APL to Python Code Generator."""

from typing import Union
from .parser import (
    Node, Scalar, Array, Name, Monad, Dyad, MonadicOp, DyadicOp,
    Assignment, Index, Function, Dfn, Guard, parse
)


class PythonGenerator:
    """Generate Python code from APL AST."""

    # Monadic function mappings
    MONADIC_MAP = {
        '+': 'apl_conjugate',
        '-': 'np.negative',
        '×': 'apl_signum',
        '÷': 'apl_reciprocal',
        '⌈': 'np.ceil',
        '⌊': 'np.floor',
        '|': 'np.abs',
        '*': 'np.exp',
        '⍟': 'np.log',
        '○': 'apl_pi_times',
        '~': 'apl_not',
        '?': 'apl_roll',
        '⍳': 'apl_iota',
        '⍴': 'apl_shape',
        ',': 'apl_ravel',
        '⌽': 'np.flip',
        '⍉': 'np.transpose',
        '⊂': 'apl_enclose',
        '⊃': 'apl_disclose',
        '⍋': 'apl_grade_up',
        '⍒': 'apl_grade_down',
        '∪': 'np.unique',
        '≢': 'apl_tally',
        '⌹': 'np.linalg.inv',
        '!': 'apl_factorial',
    }

    # Dyadic function mappings
    DYADIC_MAP = {
        '+': 'np.add',
        '-': 'np.subtract',
        '×': 'np.multiply',
        '÷': 'np.divide',
        '⌈': 'np.maximum',
        '⌊': 'np.minimum',
        '*': 'np.power',
        '⍟': 'apl_log',
        '|': 'apl_residue',
        '○': 'apl_circular',
        '<': 'apl_less',
        '≤': 'apl_less_equal',
        '=': 'apl_equal',
        '≥': 'apl_greater_equal',
        '>': 'apl_greater',
        '≠': 'apl_not_equal',
        '∧': 'apl_and',
        '∨': 'apl_or',
        '⍲': 'apl_nand',
        '⍱': 'apl_nor',
        '⍳': 'apl_index_of',
        '⍴': 'apl_reshape',
        ',': 'apl_catenate',
        '⌽': 'apl_rotate',
        '↑': 'apl_take',
        '↓': 'apl_drop',
        '∊': 'apl_membership',
        '∪': 'apl_union',
        '∩': 'apl_intersection',
        '~': 'apl_without',
        '≡': 'apl_match',
        '⊤': 'apl_encode',
        '⊥': 'apl_decode',
        '⌹': 'np.linalg.solve',
        '!': 'apl_binomial',
    }

    # Reduce operator mappings - use apl_reduce with ASCII function names
    REDUCE_MAP = {
        '+': ('apl_reduce', "'add'"),
        '-': ('apl_reduce', "'sub'"),
        '×': ('apl_reduce', "'mul'"),
        '÷': ('apl_reduce', "'div'"),
        '⌈': ('apl_reduce', "'max'"),
        '⌊': ('apl_reduce', "'min'"),
        '∧': ('apl_reduce', "'and'"),
        '∨': ('apl_reduce', "'or'"),
    }

    # Scan operator mappings
    SCAN_MAP = {
        '+': 'np.cumsum',
        '×': 'np.cumprod',
        '⌈': 'np.maximum.accumulate',
        '⌊': 'np.minimum.accumulate',
    }

    def __init__(self):
        self.indent_level = 0
        self.used_functions = set()

    def generate(self, nodes: list[Node]) -> str:
        """Generate Python code from AST nodes."""
        self.used_functions = set()

        lines = []
        for node in nodes:
            code = self._generate_node(node)
            if code:
                lines.append(code)

        # Build imports
        imports = ['import numpy as np']
        if self.used_functions:
            runtime_funcs = sorted(self.used_functions)
            imports.append(f"from ans2py.runtime import {', '.join(runtime_funcs)}")

        return '\n'.join(imports) + '\n\n' + '\n'.join(lines)

    def _generate_node(self, node: Node) -> str:
        """Generate code for a single node."""
        if node is None:
            return ''

        if isinstance(node, Scalar):
            return self._generate_scalar(node)
        elif isinstance(node, Array):
            return self._generate_array(node)
        elif isinstance(node, Name):
            return self._generate_name(node)
        elif isinstance(node, Monad):
            return self._generate_monad(node)
        elif isinstance(node, Dyad):
            return self._generate_dyad(node)
        elif isinstance(node, MonadicOp):
            return self._generate_monadic_op(node)
        elif isinstance(node, DyadicOp):
            return self._generate_dyadic_op(node)
        elif isinstance(node, Assignment):
            return self._generate_assignment(node)
        elif isinstance(node, Index):
            return self._generate_index(node)
        elif isinstance(node, Function):
            return self._generate_function(node)
        elif isinstance(node, Dfn):
            return self._generate_dfn(node)
        else:
            return f"# Unknown node: {type(node)}"

    def _generate_scalar(self, node: Scalar) -> str:
        """Generate code for a scalar value."""
        if isinstance(node.value, complex):
            return f"complex({node.value.real}, {node.value.imag})"
        elif isinstance(node.value, str):
            return repr(node.value)
        else:
            return str(node.value)

    def _generate_array(self, node: Array) -> str:
        """Generate code for an array literal."""
        elements = [self._generate_node(e) for e in node.elements]
        return f"np.array([{', '.join(elements)}])"

    def _generate_name(self, node: Name) -> str:
        """Generate code for a name reference."""
        name = node.name
        # Map APL special names
        if name == '⍵':
            return 'omega'
        elif name == '⍺':
            return 'alpha'
        elif name.startswith('⎕'):
            # System variable
            return f"apl_system_{name[1:].lower()}"
        return name

    def _generate_monad(self, node: Monad) -> str:
        """Generate code for monadic function application."""
        fn = node.fn
        right = self._generate_node(node.right)

        if fn in self.MONADIC_MAP:
            func_name = self.MONADIC_MAP[fn]
            if func_name.startswith('apl_'):
                self.used_functions.add(func_name)
            return f"{func_name}({right})"
        else:
            # User-defined function
            return f"{fn}({right})"

    def _generate_dyad(self, node: Dyad) -> str:
        """Generate code for dyadic function application."""
        fn = node.fn
        left = self._generate_node(node.left)
        right = self._generate_node(node.right)

        if fn in self.DYADIC_MAP:
            func_name = self.DYADIC_MAP[fn]
            if func_name.startswith('apl_'):
                self.used_functions.add(func_name)
            return f"{func_name}({left}, {right})"
        else:
            # User-defined function
            return f"{fn}({left}, {right})"

    def _generate_monadic_op(self, node: MonadicOp) -> str:
        """Generate code for monadic operator (e.g., +/)."""
        op = node.op
        fn = node.fn
        right = self._generate_node(node.right)

        # Reduce operator /
        if op == '/':
            if fn in self.REDUCE_MAP:
                func_name, fn_arg = self.REDUCE_MAP[fn]
                self.used_functions.add(func_name)
                return f"{func_name}({fn_arg}, {right})"
            else:
                self.used_functions.add('apl_reduce')
                return f"apl_reduce('{fn}', {right})"

        # Scan operator \
        elif op == '\\':
            if fn in self.SCAN_MAP:
                func = self.SCAN_MAP[fn]
                return f"{func}({right})"
            else:
                self.used_functions.add('apl_scan')
                return f"apl_scan('{fn}', {right})"

        # Each operator ¨
        elif op == '¨':
            if fn in self.MONADIC_MAP:
                func = self.MONADIC_MAP[fn]
                if func.startswith('apl_'):
                    self.used_functions.add(func)
                return f"np.vectorize({func})({right})"
            return f"np.vectorize({fn})({right})"

        # Commute ⍨
        elif op == '⍨':
            # Monadic ⍨ means selfie: fn⍨ x = x fn x
            if fn in self.DYADIC_MAP:
                func = self.DYADIC_MAP[fn]
                if func.startswith('apl_'):
                    self.used_functions.add(func)
                return f"{func}({right}, {right})"
            return f"{fn}({right}, {right})"

        # Inner product fn1.fn2
        elif '.' in op:
            parts = op.split('.')
            if len(parts) == 2:
                fn1, fn2 = parts
                self.used_functions.add('apl_inner_product')
                return f"apl_inner_product('{fn1}', '{fn2}', {right})"

        return f"# Unknown operator: {op} {fn}"

    def _generate_dyadic_op(self, node: DyadicOp) -> str:
        """Generate code for dyadic operator."""
        op = node.op
        fn1 = node.fn1
        fn2 = node.fn2
        left = self._generate_node(node.left)
        right = self._generate_node(node.right)

        # Inner product
        if op == '.':
            if fn1 == '+' and fn2 == '×':
                return f"np.dot({left}, {right})"
            self.used_functions.add('apl_inner_product')
            return f"apl_inner_product('{fn1}', '{fn2}', {left}, {right})"

        # Outer product ∘.
        elif op == '∘.':
            self.used_functions.add('apl_outer_product')
            return f"apl_outer_product('{fn2}', {left}, {right})"

        # Compress / as dyadic
        elif op == '/':
            self.used_functions.add('apl_compress')
            return f"apl_compress({left}, {right})"

        # Expand \ as dyadic
        elif op == '\\':
            self.used_functions.add('apl_expand')
            return f"apl_expand({left}, {right})"

        return f"# Unknown dyadic operator: {op}"

    def _generate_assignment(self, node: Assignment) -> str:
        """Generate code for assignment."""
        name = node.name
        value = self._generate_node(node.value)
        return f"{name} = {value}"

    def _generate_index(self, node: Index) -> str:
        """Generate code for array indexing."""
        array = self._generate_node(node.array) if node.array else ""
        indices = []
        for idx in node.indices:
            if idx is None:
                indices.append(':')
            else:
                # APL is 1-indexed, Python is 0-indexed
                idx_code = self._generate_node(idx)
                indices.append(f"({idx_code}) - 1")

        if len(indices) == 1:
            return f"{array}[{indices[0]}]"
        return f"{array}[{', '.join(indices)}]"

    def _generate_function(self, node: Function) -> str:
        """Generate code for user-defined function."""
        params = []
        if node.left_arg:
            params.append(node.left_arg.lower())
        if node.right_arg:
            params.append(node.right_arg.lower())
        if not params:
            params = ['omega']

        param_str = ', '.join(params)
        body_lines = [self._generate_node(stmt) for stmt in node.body]
        body = '\n    '.join(body_lines) if body_lines else 'pass'

        result_var = node.result.lower() if node.result else 'result'

        return f"""def {node.name.lower()}({param_str}):
    {body}
    return {result_var}"""

    def _generate_dfn(self, node: Dfn) -> str:
        """Generate code for direct function."""
        if len(node.body) == 1:
            # Single expression - can be lambda
            body = self._generate_node(node.body[0])
            return f"lambda omega, alpha=None: {body}"
        else:
            body_lines = [self._generate_node(stmt) for stmt in node.body]
            body = '\n    '.join(body_lines)
            return f"""def _dfn(omega, alpha=None):
    {body}"""


def generate_python(source: str) -> str:
    """Generate Python code from APL source."""
    ast = parse(source)
    generator = PythonGenerator()
    return generator.generate(ast)


def convert_apl_to_python(apl_code: str, function_name: str = None) -> str:
    """Convert APL code to Python, optionally wrapping in a function."""
    ast = parse(apl_code)
    generator = PythonGenerator()

    if function_name:
        # Wrap in a function
        code = generator.generate(ast)
        # Extract the body (skip imports)
        lines = code.split('\n')
        import_lines = [l for l in lines if l.startswith('import') or l.startswith('from')]
        body_lines = [l for l in lines if not l.startswith('import') and not l.startswith('from') and l.strip()]

        return '\n'.join(import_lines) + f'\n\ndef {function_name}(omega, alpha=None):\n    ' + '\n    '.join(body_lines) + '\n    return result'
    else:
        return generator.generate(ast)
