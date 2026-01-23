"""APL Runtime Library - Helper functions for APL operations."""

import numpy as np
from typing import Union

# Type aliases
ArrayLike = Union[np.ndarray, list, int, float, complex]


def ensure_array(x: ArrayLike) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    return np.atleast_1d(np.array(x))


# ============================================================
# Monadic Scalar Functions
# ============================================================

def apl_conjugate(x: ArrayLike) -> np.ndarray:
    """+ monadic: conjugate (identity for real)"""
    return np.conjugate(ensure_array(x))


def apl_negate(x: ArrayLike) -> np.ndarray:
    """- monadic: negate"""
    return np.negative(ensure_array(x))


def apl_signum(x: ArrayLike) -> np.ndarray:
    """× monadic: signum/direction"""
    arr = ensure_array(x)
    if np.issubdtype(arr.dtype, np.complexfloating):
        return np.where(arr == 0, 0, arr / np.abs(arr))
    return np.sign(arr)


def apl_reciprocal(x: ArrayLike) -> np.ndarray:
    """÷ monadic: reciprocal"""
    return 1.0 / ensure_array(x)


def apl_ceiling(x: ArrayLike) -> np.ndarray:
    """⌈ monadic: ceiling"""
    return np.ceil(ensure_array(x))


def apl_floor(x: ArrayLike) -> np.ndarray:
    """⌊ monadic: floor"""
    return np.floor(ensure_array(x))


def apl_magnitude(x: ArrayLike) -> np.ndarray:
    """| monadic: magnitude/absolute value"""
    return np.abs(ensure_array(x))


def apl_exponential(x: ArrayLike) -> np.ndarray:
    """* monadic: exponential (e^x)"""
    return np.exp(ensure_array(x))


def apl_natural_log(x: ArrayLike) -> np.ndarray:
    """⍟ monadic: natural logarithm"""
    return np.log(ensure_array(x))


def apl_factorial(x: ArrayLike) -> np.ndarray:
    """! monadic: factorial"""
    from scipy.special import factorial
    return factorial(ensure_array(x), exact=False)


def apl_pi_times(x: ArrayLike) -> np.ndarray:
    """○ monadic: pi times"""
    return np.pi * ensure_array(x)


def apl_not(x: ArrayLike) -> np.ndarray:
    """~ monadic: logical not"""
    return 1 - ensure_array(x)


def apl_roll(x: ArrayLike) -> np.ndarray:
    """? monadic: roll (random integers 1 to x)"""
    arr = ensure_array(x).astype(int)
    return np.array([np.random.randint(1, n + 1) if n > 0 else 0 for n in arr.flat]).reshape(arr.shape)


# ============================================================
# Dyadic Scalar Functions
# ============================================================

def apl_add(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """+ dyadic: add"""
    return np.add(ensure_array(a), ensure_array(b))


def apl_subtract(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """- dyadic: subtract"""
    return np.subtract(ensure_array(a), ensure_array(b))


def apl_multiply(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """× dyadic: multiply"""
    return np.multiply(ensure_array(a), ensure_array(b))


def apl_divide(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """÷ dyadic: divide"""
    return np.divide(ensure_array(a), ensure_array(b))


def apl_maximum(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⌈ dyadic: maximum"""
    return np.maximum(ensure_array(a), ensure_array(b))


def apl_minimum(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⌊ dyadic: minimum"""
    return np.minimum(ensure_array(a), ensure_array(b))


def apl_power(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """* dyadic: power"""
    return np.power(ensure_array(a), ensure_array(b))


def apl_log(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⍟ dyadic: logarithm (base a of b)"""
    return np.log(ensure_array(b)) / np.log(ensure_array(a))


def apl_residue(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """| dyadic: residue/modulo"""
    return np.mod(ensure_array(b), ensure_array(a))


def apl_binomial(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """! dyadic: binomial coefficient"""
    from scipy.special import comb
    return comb(ensure_array(a), ensure_array(b), exact=False)


def apl_circular(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """○ dyadic: circular/trigonometric functions"""
    a = ensure_array(a)
    b = ensure_array(b)
    result = np.zeros_like(b, dtype=float)

    funcs = {
        0: lambda x: np.sqrt(1 - x**2),
        1: np.sin,
        2: np.cos,
        3: np.tan,
        4: lambda x: np.sqrt(1 + x**2),
        5: np.sinh,
        6: np.cosh,
        7: np.tanh,
        -1: np.arcsin,
        -2: np.arccos,
        -3: np.arctan,
        -4: lambda x: np.sqrt(x**2 - 1),
        -5: np.arcsinh,
        -6: np.arccosh,
        -7: np.arctanh,
    }

    for i, (ai, bi) in enumerate(zip(a.flat, b.flat)):
        if int(ai) in funcs:
            result.flat[i] = funcs[int(ai)](bi)

    return result


# ============================================================
# Comparison Functions
# ============================================================

def apl_less(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """< dyadic: less than"""
    return (ensure_array(a) < ensure_array(b)).astype(int)


def apl_less_equal(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """≤ dyadic: less than or equal"""
    return (ensure_array(a) <= ensure_array(b)).astype(int)


def apl_equal(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """= dyadic: equal"""
    return (ensure_array(a) == ensure_array(b)).astype(int)


def apl_greater_equal(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """≥ dyadic: greater than or equal"""
    return (ensure_array(a) >= ensure_array(b)).astype(int)


def apl_greater(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """> dyadic: greater than"""
    return (ensure_array(a) > ensure_array(b)).astype(int)


def apl_not_equal(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """≠ dyadic: not equal"""
    return (ensure_array(a) != ensure_array(b)).astype(int)


# ============================================================
# Logical Functions
# ============================================================

def apl_and(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """∧ dyadic: logical and (or LCM for integers)"""
    a, b = ensure_array(a), ensure_array(b)
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        return np.lcm(a, b)
    return np.logical_and(a, b).astype(int)


def apl_or(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """∨ dyadic: logical or (or GCD for integers)"""
    a, b = ensure_array(a), ensure_array(b)
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        return np.gcd(a, b)
    return np.logical_or(a, b).astype(int)


def apl_nand(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⍲ dyadic: nand"""
    return 1 - apl_and(a, b)


def apl_nor(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⍱ dyadic: nor"""
    return 1 - apl_or(a, b)


# ============================================================
# Array Functions
# ============================================================

def apl_iota(n: ArrayLike) -> np.ndarray:
    """⍳ monadic: index generator (1-based)"""
    n = int(np.array(n).flat[0])
    return np.arange(1, n + 1)


def apl_index_of(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⍳ dyadic: index of (find b in a)"""
    a, b = ensure_array(a), ensure_array(b)
    result = np.zeros(b.shape, dtype=int)
    for i, val in enumerate(b.flat):
        idx = np.where(a == val)[0]
        result.flat[i] = idx[0] + 1 if len(idx) > 0 else len(a) + 1
    return result


def apl_shape(x: ArrayLike) -> np.ndarray:
    """⍴ monadic: shape"""
    return np.array(ensure_array(x).shape)


def apl_reshape(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⍴ dyadic: reshape"""
    a = ensure_array(a).astype(int)
    b = ensure_array(b)
    shape = tuple(a.flat)
    size = int(np.prod(shape))
    flat = np.resize(b.flatten(), size)
    return flat.reshape(shape)


def apl_ravel(x: ArrayLike) -> np.ndarray:
    """, monadic: ravel (flatten)"""
    return ensure_array(x).flatten()


def apl_catenate(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """, dyadic: catenate"""
    a, b = ensure_array(a), ensure_array(b)
    return np.concatenate([a.flatten(), b.flatten()])


def apl_reverse(x: ArrayLike) -> np.ndarray:
    """⌽ monadic: reverse"""
    return np.flip(ensure_array(x))


def apl_rotate(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⌽ dyadic: rotate"""
    a = int(np.array(a).flat[0])
    return np.roll(ensure_array(b), -a)


def apl_transpose(x: ArrayLike) -> np.ndarray:
    """⍉ monadic: transpose"""
    arr = ensure_array(x)
    if arr.ndim < 2:
        return arr
    return np.transpose(arr)


def apl_take(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """↑ dyadic: take"""
    n = int(np.array(a).flat[0])
    arr = ensure_array(b).flatten()
    if n >= 0:
        if n <= len(arr):
            return arr[:n]
        else:
            return np.pad(arr, (0, n - len(arr)), mode='constant')
    else:
        n = -n
        if n <= len(arr):
            return arr[-n:]
        else:
            return np.pad(arr, (n - len(arr), 0), mode='constant')


def apl_drop(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """↓ dyadic: drop"""
    n = int(np.array(a).flat[0])
    arr = ensure_array(b).flatten()
    if n >= 0:
        return arr[n:]
    else:
        return arr[:n]


def apl_enclose(x: ArrayLike) -> np.ndarray:
    """⊂ monadic: enclose"""
    return np.array([ensure_array(x)], dtype=object)


def apl_disclose(x: ArrayLike) -> np.ndarray:
    """⊃ monadic: disclose/first"""
    arr = ensure_array(x)
    if arr.dtype == object and len(arr) > 0:
        return arr[0]
    return arr.flat[0] if arr.size > 0 else arr


def apl_membership(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """∊ dyadic: membership"""
    a, b = ensure_array(a), ensure_array(b)
    return np.isin(a, b).astype(int)


def apl_grade_up(x: ArrayLike) -> np.ndarray:
    """⍋ monadic: grade up (indices that would sort ascending)"""
    return np.argsort(ensure_array(x).flatten()) + 1


def apl_grade_down(x: ArrayLike) -> np.ndarray:
    """⍒ monadic: grade down (indices that would sort descending)"""
    return np.argsort(-ensure_array(x).flatten()) + 1


def apl_unique(x: ArrayLike) -> np.ndarray:
    """∪ monadic: unique"""
    return np.unique(ensure_array(x))


def apl_union(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """∪ dyadic: union"""
    return np.union1d(ensure_array(a), ensure_array(b))


def apl_intersection(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """∩ dyadic: intersection"""
    return np.intersect1d(ensure_array(a), ensure_array(b))


def apl_without(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """~ dyadic: without"""
    return np.setdiff1d(ensure_array(a), ensure_array(b))


def apl_tally(x: ArrayLike) -> int:
    """≢ monadic: tally (count of first dimension)"""
    arr = ensure_array(x)
    return arr.shape[0] if arr.ndim > 0 else 1


def apl_match(a: ArrayLike, b: ArrayLike) -> int:
    """≡ dyadic: match"""
    return int(np.array_equal(ensure_array(a), ensure_array(b)))


# ============================================================
# Reduce and Scan Operations
# ============================================================

def apl_reduce(fn: str, x: ArrayLike, axis: int = -1) -> np.ndarray:
    """Apply reduction with given function along the last axis (APL default)."""
    arr = ensure_array(x)

    # For 1D arrays, reduce over the entire array
    if arr.ndim <= 1:
        axis = None

    reductions = {
        # ASCII names (preferred for generated code)
        'add': lambda a: np.sum(a, axis=axis),
        'sub': lambda a: np.subtract.reduce(a, axis=axis),
        'mul': lambda a: np.prod(a, axis=axis),
        'div': lambda a: np.divide.reduce(a, axis=axis),
        'max': lambda a: np.max(a, axis=axis),
        'min': lambda a: np.min(a, axis=axis),
        'and': lambda a: np.logical_and.reduce(a, axis=axis).astype(int),
        'or': lambda a: np.logical_or.reduce(a, axis=axis).astype(int),
        # APL symbols (for backwards compatibility)
        '+': lambda a: np.sum(a, axis=axis),
        '-': lambda a: np.subtract.reduce(a, axis=axis),
        '×': lambda a: np.prod(a, axis=axis),
        '÷': lambda a: np.divide.reduce(a, axis=axis),
        '⌈': lambda a: np.max(a, axis=axis),
        '⌊': lambda a: np.min(a, axis=axis),
        '∧': lambda a: np.logical_and.reduce(a, axis=axis).astype(int),
        '∨': lambda a: np.logical_or.reduce(a, axis=axis).astype(int),
    }

    if fn in reductions:
        return reductions[fn](arr)
    return arr


def apl_scan(fn: str, x: ArrayLike) -> np.ndarray:
    """Apply scan with given function."""
    arr = ensure_array(x)

    scans = {
        '+': np.cumsum,
        '×': np.cumprod,
        '⌈': np.maximum.accumulate,
        '⌊': np.minimum.accumulate,
    }

    if fn in scans:
        return scans[fn](arr)
    return arr


def apl_compress(mask: ArrayLike, x: ArrayLike) -> np.ndarray:
    """/ dyadic: compress (boolean selection)"""
    mask = ensure_array(mask).astype(bool)
    arr = ensure_array(x)
    return arr[mask]


def apl_expand(mask: ArrayLike, x: ArrayLike) -> np.ndarray:
    """\\ dyadic: expand"""
    mask = ensure_array(mask).astype(bool)
    arr = ensure_array(x).flatten()
    result = np.zeros(len(mask), dtype=arr.dtype)
    result[mask] = arr[:mask.sum()]
    return result


# ============================================================
# Matrix Operations
# ============================================================

def apl_matrix_inverse(x: ArrayLike) -> np.ndarray:
    """⌹ monadic: matrix inverse"""
    return np.linalg.inv(ensure_array(x))


def apl_matrix_divide(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⌹ dyadic: matrix divide (solve linear system)"""
    return np.linalg.solve(ensure_array(a), ensure_array(b))


def apl_inner_product(fn1: str, fn2: str, a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Inner product: a fn1.fn2 b"""
    a, b = ensure_array(a), ensure_array(b)

    if fn1 == '+' and fn2 == '×':
        return np.dot(a, b)

    # General inner product
    if a.ndim == 1 and b.ndim == 1:
        paired = apply_dyadic(fn2, a, b)
        return apl_reduce(fn1, paired)

    return np.dot(a, b)


def apl_outer_product(fn: str, a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Outer product: a ∘.fn b"""
    a, b = ensure_array(a), ensure_array(b)

    outer_funcs = {
        '+': np.add.outer,
        '-': np.subtract.outer,
        '×': np.multiply.outer,
        '÷': np.divide.outer,
        '⌈': np.maximum.outer,
        '⌊': np.minimum.outer,
        '=': lambda x, y: np.equal.outer(x, y).astype(int),
        '≠': lambda x, y: np.not_equal.outer(x, y).astype(int),
        '<': lambda x, y: np.less.outer(x, y).astype(int),
        '≤': lambda x, y: np.less_equal.outer(x, y).astype(int),
        '>': lambda x, y: np.greater.outer(x, y).astype(int),
        '≥': lambda x, y: np.greater_equal.outer(x, y).astype(int),
    }

    if fn in outer_funcs:
        return outer_funcs[fn](a.flatten(), b.flatten())

    return np.outer(a, b)


# ============================================================
# Encode/Decode
# ============================================================

def apl_encode(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⊤ dyadic: encode (representation)"""
    a = ensure_array(a).astype(int)
    b = int(ensure_array(b).flat[0])

    result = []
    for base in reversed(a.flat):
        if base == 0:
            result.append(b)
            b = 0
        else:
            result.append(b % base)
            b //= base
    return np.array(list(reversed(result)))


def apl_decode(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """⊥ dyadic: decode (base value)"""
    a = ensure_array(a).astype(int)
    b = ensure_array(b).astype(int)
    return np.polyval(b, a[0]) if len(a) == 1 else int(np.dot(np.cumprod([1] + list(a[:-1])), b))


# ============================================================
# Helper function dispatcher
# ============================================================

MONADIC_FUNCS = {
    '+': apl_conjugate,
    '-': apl_negate,
    '×': apl_signum,
    '÷': apl_reciprocal,
    '⌈': apl_ceiling,
    '⌊': apl_floor,
    '|': apl_magnitude,
    '*': apl_exponential,
    '⍟': apl_natural_log,
    '○': apl_pi_times,
    '~': apl_not,
    '?': apl_roll,
    '⍳': apl_iota,
    '⍴': apl_shape,
    ',': apl_ravel,
    '⌽': apl_reverse,
    '⍉': apl_transpose,
    '⊂': apl_enclose,
    '⊃': apl_disclose,
    '⍋': apl_grade_up,
    '⍒': apl_grade_down,
    '∪': apl_unique,
    '≢': apl_tally,
    '⌹': apl_matrix_inverse,
}

DYADIC_FUNCS = {
    '+': apl_add,
    '-': apl_subtract,
    '×': apl_multiply,
    '÷': apl_divide,
    '⌈': apl_maximum,
    '⌊': apl_minimum,
    '*': apl_power,
    '⍟': apl_log,
    '|': apl_residue,
    '○': apl_circular,
    '<': apl_less,
    '≤': apl_less_equal,
    '=': apl_equal,
    '≥': apl_greater_equal,
    '>': apl_greater,
    '≠': apl_not_equal,
    '∧': apl_and,
    '∨': apl_or,
    '⍲': apl_nand,
    '⍱': apl_nor,
    '⍳': apl_index_of,
    '⍴': apl_reshape,
    ',': apl_catenate,
    '⌽': apl_rotate,
    '↑': apl_take,
    '↓': apl_drop,
    '∊': apl_membership,
    '∪': apl_union,
    '∩': apl_intersection,
    '~': apl_without,
    '≡': apl_match,
    '⊤': apl_encode,
    '⊥': apl_decode,
    '⌹': apl_matrix_divide,
}


def apply_monadic(fn: str, x: ArrayLike) -> np.ndarray:
    """Apply monadic function."""
    if fn in MONADIC_FUNCS:
        return MONADIC_FUNCS[fn](x)
    raise ValueError(f"Unknown monadic function: {fn}")


def apply_dyadic(fn: str, a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Apply dyadic function."""
    if fn in DYADIC_FUNCS:
        return DYADIC_FUNCS[fn](a, b)
    raise ValueError(f"Unknown dyadic function: {fn}")
