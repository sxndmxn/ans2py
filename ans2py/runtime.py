"""APL Runtime Library - NumPy-based APL primitives (no external dependencies beyond NumPy)."""

import numpy as np
from typing import Union, Callable
from functools import reduce as functools_reduce

# Type aliases
ArrayLike = Union[np.ndarray, list, int, float, complex]


def ensure_array(x: ArrayLike) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    return np.atleast_1d(np.array(x))


def scalar_result(x: np.ndarray):
    """Return scalar if single element, else array."""
    return x.item() if x.size == 1 else x


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
    """x monadic: signum/direction"""
    arr = ensure_array(x)
    if np.issubdtype(arr.dtype, np.complexfloating):
        return np.where(arr == 0, 0, arr / np.abs(arr))
    return np.sign(arr)


def apl_reciprocal(x: ArrayLike) -> np.ndarray:
    """div monadic: reciprocal"""
    return 1.0 / ensure_array(x)


def apl_ceiling(x: ArrayLike) -> np.ndarray:
    """ceil monadic: ceiling"""
    return np.ceil(ensure_array(x)).astype(int)


def apl_floor(x: ArrayLike) -> np.ndarray:
    """floor monadic: floor"""
    return np.floor(ensure_array(x)).astype(int)


def apl_magnitude(x: ArrayLike) -> np.ndarray:
    """| monadic: magnitude/absolute value"""
    return np.abs(ensure_array(x))


def apl_exponential(x: ArrayLike) -> np.ndarray:
    """* monadic: exponential (e^x)"""
    return np.exp(ensure_array(x))


def apl_natural_log(x: ArrayLike) -> np.ndarray:
    """log monadic: natural logarithm"""
    return np.log(ensure_array(x))


def apl_factorial(x: ArrayLike) -> np.ndarray:
    """! monadic: factorial (using gamma function, no scipy needed)"""
    arr = ensure_array(x)
    # For integers, use iterative factorial; for floats, use gamma approximation
    def fact(n):
        if n < 0:
            return np.nan
        if n == 0 or n == 1:
            return 1
        if isinstance(n, (int, np.integer)) and n < 20:
            result = 1
            for i in range(2, int(n) + 1):
                result *= i
            return result
        # Stirling's approximation for large n or floats
        return np.sqrt(2 * np.pi * n) * (n / np.e) ** n

    return np.vectorize(fact)(arr)


def apl_pi_times(x: ArrayLike) -> np.ndarray:
    """o monadic: pi times"""
    return np.pi * ensure_array(x)


def apl_not(x: ArrayLike) -> np.ndarray:
    """~ monadic: logical not"""
    return (1 - ensure_array(x)).astype(int)


def apl_roll(x: ArrayLike) -> np.ndarray:
    """? monadic: roll (random integers 1 to x)"""
    arr = ensure_array(x).astype(int)
    return np.array([np.random.randint(1, n + 1) if n > 0 else 0 for n in arr.flat]).reshape(arr.shape)


def apl_where(x: ArrayLike) -> np.ndarray:
    """where monadic: indices where 1"""
    arr = ensure_array(x).astype(bool)
    return np.nonzero(arr.flatten())[0] + 1  # 1-based


def apl_format(x: ArrayLike) -> str:
    """format monadic: convert to string"""
    arr = ensure_array(x)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.ndim == 1:
        return ' '.join(str(v) for v in arr)
    return '\n'.join(' '.join(str(v) for v in row) for row in arr)


def apl_execute(x: str) -> np.ndarray:
    """execute monadic: evaluate string (limited - just numeric)"""
    try:
        return np.array(eval(x))
    except:
        return np.array([])


def apl_depth(x: ArrayLike) -> int:
    """depth monadic: nesting depth"""
    arr = ensure_array(x)
    if arr.dtype != object:
        return 0 if arr.ndim == 0 else 1
    max_depth = 0
    for item in arr.flat:
        if isinstance(item, np.ndarray):
            max_depth = max(max_depth, 1 + apl_depth(item))
        else:
            max_depth = max(max_depth, 1)
    return max_depth


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
    """x dyadic: multiply"""
    return np.multiply(ensure_array(a), ensure_array(b))


def apl_divide(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """div dyadic: divide"""
    return np.divide(ensure_array(a), ensure_array(b))


def apl_maximum(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """ceil dyadic: maximum"""
    return np.maximum(ensure_array(a), ensure_array(b))


def apl_minimum(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """floor dyadic: minimum"""
    return np.minimum(ensure_array(a), ensure_array(b))


def apl_power(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """* dyadic: power"""
    return np.power(ensure_array(a), ensure_array(b))


def apl_log(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """log dyadic: logarithm (base a of b)"""
    return np.log(ensure_array(b)) / np.log(ensure_array(a))


def apl_residue(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """| dyadic: residue/modulo"""
    a_arr = ensure_array(a)
    b_arr = ensure_array(b)
    # APL residue: result has same sign as a (divisor)
    return np.where(a_arr == 0, b_arr, np.mod(b_arr, a_arr))


def apl_binomial(n: ArrayLike, k: ArrayLike) -> np.ndarray:
    """! dyadic: binomial coefficient (n choose k)"""
    n_arr = ensure_array(n)
    k_arr = ensure_array(k)

    def binom(n, k):
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    return np.vectorize(binom)(n_arr, k_arr)


def apl_circular(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """o dyadic: circular/trigonometric functions"""
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


def apl_left(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """left dyadic: return left argument"""
    return ensure_array(a)


def apl_right(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """right dyadic: return right argument"""
    return ensure_array(b)


def apl_left_monadic(x: ArrayLike) -> np.ndarray:
    """left monadic: same (identity)"""
    return ensure_array(x)


def apl_right_monadic(x: ArrayLike) -> np.ndarray:
    """right monadic: same (identity)"""
    return ensure_array(x)


# Alias for same/identity
apl_same = apl_right_monadic


# ============================================================
# Comparison Functions
# ============================================================

def apl_less(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """< dyadic: less than"""
    return (ensure_array(a) < ensure_array(b)).astype(int)


def apl_less_equal(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """le dyadic: less than or equal"""
    return (ensure_array(a) <= ensure_array(b)).astype(int)


def apl_equal(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """= dyadic: equal"""
    return (ensure_array(a) == ensure_array(b)).astype(int)


def apl_greater_equal(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """ge dyadic: greater than or equal"""
    return (ensure_array(a) >= ensure_array(b)).astype(int)


def apl_greater(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """> dyadic: greater than"""
    return (ensure_array(a) > ensure_array(b)).astype(int)


def apl_not_equal(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """ne dyadic: not equal"""
    return (ensure_array(a) != ensure_array(b)).astype(int)


# ============================================================
# Logical Functions
# ============================================================

def apl_and(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """and dyadic: logical and (or LCM for non-boolean)"""
    a, b = ensure_array(a), ensure_array(b)
    # Check if boolean-like (0s and 1s only)
    if np.all((a == 0) | (a == 1)) and np.all((b == 0) | (b == 1)):
        return np.logical_and(a, b).astype(int)
    # Otherwise LCM
    return np.lcm(a.astype(int), b.astype(int))


def apl_or(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """or dyadic: logical or (or GCD for non-boolean)"""
    a, b = ensure_array(a), ensure_array(b)
    if np.all((a == 0) | (a == 1)) and np.all((b == 0) | (b == 1)):
        return np.logical_or(a, b).astype(int)
    return np.gcd(a.astype(int), b.astype(int))


def apl_nand(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """nand dyadic: nand"""
    return 1 - np.logical_and(ensure_array(a), ensure_array(b)).astype(int)


def apl_nor(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """nor dyadic: nor"""
    return 1 - np.logical_or(ensure_array(a), ensure_array(b)).astype(int)


# ============================================================
# Array Functions
# ============================================================

def apl_iota(n: ArrayLike) -> np.ndarray:
    """iota monadic: index generator (1-based)"""
    arr = ensure_array(n)
    if arr.size == 1:
        return np.arange(1, int(arr.flat[0]) + 1)
    # Multi-dimensional iota
    shape = tuple(arr.astype(int).flatten())
    size = int(np.prod(shape))
    return np.arange(1, size + 1).reshape(shape)


def apl_index_of(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """iota dyadic: index of (find b in a)"""
    a, b = ensure_array(a), ensure_array(b)
    a_flat = a.flatten()
    result = np.zeros(b.shape, dtype=int)
    for i, val in enumerate(b.flat):
        idx = np.where(a_flat == val)[0]
        result.flat[i] = idx[0] + 1 if len(idx) > 0 else len(a_flat) + 1
    return result


def apl_shape(x: ArrayLike) -> np.ndarray:
    """shape monadic: shape"""
    return np.array(ensure_array(x).shape)


def apl_reshape(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """reshape dyadic: reshape"""
    a = ensure_array(a).astype(int)
    b = ensure_array(b)
    shape = tuple(a.flat)
    size = int(np.prod(shape))
    flat = np.resize(b.flatten(), size)
    return flat.reshape(shape)


def apl_ravel(x: ArrayLike) -> np.ndarray:
    """ravel monadic: ravel (flatten)"""
    return ensure_array(x).flatten()


def apl_catenate(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """catenate dyadic: catenate"""
    a, b = ensure_array(a), ensure_array(b)
    if a.ndim == 1 and b.ndim == 1:
        return np.concatenate([a, b])
    # Match dimensions for catenation
    if a.ndim != b.ndim:
        if a.ndim < b.ndim:
            a = a.reshape((1,) * (b.ndim - a.ndim) + a.shape)
        else:
            b = b.reshape((1,) * (a.ndim - b.ndim) + b.shape)
    return np.concatenate([a, b], axis=-1)


def apl_catenate_first(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """table dyadic: catenate along first axis"""
    a, b = ensure_array(a), ensure_array(b)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return np.vstack([a, b])


def apl_reverse(x: ArrayLike) -> np.ndarray:
    """reverse monadic: reverse"""
    return np.flip(ensure_array(x), axis=-1)


def apl_reverse_first(x: ArrayLike) -> np.ndarray:
    """reverse_first monadic: reverse along first axis"""
    return np.flip(ensure_array(x), axis=0)


def apl_rotate(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """rotate dyadic: rotate"""
    a = int(np.array(a).flat[0])
    return np.roll(ensure_array(b), -a, axis=-1)


def apl_rotate_first(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """rotate_first dyadic: rotate along first axis"""
    a = int(np.array(a).flat[0])
    return np.roll(ensure_array(b), -a, axis=0)


def apl_transpose(x: ArrayLike) -> np.ndarray:
    """transpose monadic: transpose"""
    arr = ensure_array(x)
    if arr.ndim < 2:
        return arr
    return np.transpose(arr)


def apl_transpose_dyadic(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """transpose dyadic: dyadic transpose with axis reordering"""
    a = ensure_array(a).astype(int) - 1  # Convert to 0-based
    b = ensure_array(b)
    return np.transpose(b, axes=tuple(a.flatten()))


def apl_take(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """take dyadic: take"""
    a_arr = ensure_array(a).astype(int).flatten()
    b_arr = ensure_array(b)

    if len(a_arr) == 1:
        n = a_arr[0]
        flat = b_arr.flatten()
        if n >= 0:
            if n <= len(flat):
                return flat[:n]
            else:
                return np.pad(flat, (0, n - len(flat)), mode='constant')
        else:
            n = -n
            if n <= len(flat):
                return flat[-n:]
            else:
                return np.pad(flat, (n - len(flat), 0), mode='constant')

    # Multi-dimensional take
    result = b_arr
    for axis, n in enumerate(a_arr):
        if axis >= result.ndim:
            break
        size = result.shape[axis]
        if n >= 0:
            if n <= size:
                result = np.take(result, range(n), axis=axis)
            else:
                pad_width = [(0, 0)] * result.ndim
                pad_width[axis] = (0, n - size)
                result = np.pad(result, pad_width, mode='constant')
        else:
            n = -n
            if n <= size:
                result = np.take(result, range(size - n, size), axis=axis)
            else:
                pad_width = [(0, 0)] * result.ndim
                pad_width[axis] = (n - size, 0)
                result = np.pad(result, pad_width, mode='constant')
    return result


def apl_drop(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """drop dyadic: drop"""
    a_arr = ensure_array(a).astype(int).flatten()
    b_arr = ensure_array(b)

    if len(a_arr) == 1:
        n = a_arr[0]
        flat = b_arr.flatten()
        if n >= 0:
            return flat[n:]
        else:
            return flat[:n]

    # Multi-dimensional drop
    slices = []
    for axis, n in enumerate(a_arr):
        if axis >= b_arr.ndim:
            break
        size = b_arr.shape[axis]
        if n >= 0:
            slices.append(slice(n, None))
        else:
            slices.append(slice(None, n))

    while len(slices) < b_arr.ndim:
        slices.append(slice(None))

    return b_arr[tuple(slices)]


def apl_enclose(x: ArrayLike) -> np.ndarray:
    """enclose monadic: enclose (box)"""
    return np.array([ensure_array(x)], dtype=object)


def apl_disclose(x: ArrayLike) -> np.ndarray:
    """disclose monadic: disclose/first"""
    arr = ensure_array(x)
    if arr.dtype == object and arr.size > 0:
        # Get first element along first axis
        first = arr[0] if arr.ndim > 0 else arr.item()
        if isinstance(first, np.ndarray):
            return first
        return np.array(first)
    if arr.size > 0:
        return np.array(arr.flat[0])
    return arr


def apl_first(x: ArrayLike):
    """first monadic: first element"""
    arr = ensure_array(x)
    return arr.flat[0] if arr.size > 0 else 0


def apl_pick(a: ArrayLike, b: ArrayLike):
    """pick dyadic: pick element"""
    a_arr = ensure_array(a).astype(int) - 1  # 0-based
    b_arr = ensure_array(b)
    if a_arr.size == 1:
        return b_arr.flat[a_arr.flat[0]]
    return b_arr[tuple(a_arr.flatten())]


def apl_membership(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """membership dyadic: membership"""
    a, b = ensure_array(a), ensure_array(b)
    return np.isin(a, b).astype(int)


def apl_find(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """find dyadic: find pattern a in b"""
    a, b = ensure_array(a).flatten(), ensure_array(b).flatten()
    result = np.zeros(len(b), dtype=int)
    a_len = len(a)
    for i in range(len(b) - a_len + 1):
        if np.array_equal(b[i:i+a_len], a):
            result[i] = 1
    return result


def apl_grade_up(x: ArrayLike) -> np.ndarray:
    """grade_up monadic: grade up (indices that would sort ascending)"""
    return np.argsort(ensure_array(x).flatten()) + 1


def apl_grade_down(x: ArrayLike) -> np.ndarray:
    """grade_down monadic: grade down (indices that would sort descending)"""
    return np.argsort(-ensure_array(x).flatten()) + 1


def apl_unique(x: ArrayLike) -> np.ndarray:
    """unique monadic: unique (preserves order)"""
    arr = ensure_array(x).flatten()
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


def apl_union(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """union dyadic: union"""
    a, b = ensure_array(a).flatten(), ensure_array(b).flatten()
    return np.concatenate([a, b[~np.isin(b, a)]])


def apl_intersection(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """intersection dyadic: intersection"""
    a, b = ensure_array(a).flatten(), ensure_array(b).flatten()
    return a[np.isin(a, b)]


def apl_without(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """without dyadic: without (set difference)"""
    a, b = ensure_array(a).flatten(), ensure_array(b).flatten()
    return a[~np.isin(a, b)]


def apl_tally(x: ArrayLike) -> int:
    """tally monadic: tally (count of first dimension)"""
    arr = ensure_array(x)
    return arr.shape[0] if arr.ndim > 0 else 1


def apl_match(a: ArrayLike, b: ArrayLike) -> int:
    """match dyadic: match (deep equality)"""
    a_arr, b_arr = ensure_array(a), ensure_array(b)
    if a_arr.shape != b_arr.shape:
        return 0
    return int(np.array_equal(a_arr, b_arr))


def apl_not_match(a: ArrayLike, b: ArrayLike) -> int:
    """not_match dyadic: not match"""
    return 1 - apl_match(a, b)


def apl_index(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """index dyadic: index/squad indexing"""
    a_arr = ensure_array(a).astype(int) - 1  # 0-based
    b_arr = ensure_array(b)
    return b_arr[tuple(a_arr.flatten())]


# ============================================================
# Reduce and Scan Operations
# ============================================================

def apl_reduce(fn: str, x: ArrayLike, axis: int = -1) -> np.ndarray:
    """Apply reduction with given function along the last axis (APL default)."""
    arr = ensure_array(x)

    # For 1D arrays, reduce over the entire array
    if arr.ndim <= 1:
        axis = None

    # APL reduction is right-to-left: -/1 2 3 4 means 1-(2-(3-4)) = -2
    def right_reduce(op, a):
        """Right-to-left reduction (APL style)."""
        if axis is not None:
            # For multi-dim, apply along axis (NumPy default is left-to-right)
            return np.apply_along_axis(lambda row: functools_reduce(lambda acc, x: op(x, acc), reversed(row)), axis, a)
        return functools_reduce(lambda acc, x: op(x, acc), reversed(a.flatten()))

    reductions = {
        # ASCII names (preferred for generated code)
        'add': lambda a: np.sum(a, axis=axis),
        'sub': lambda a: right_reduce(lambda x, y: x - y, a),
        'mul': lambda a: np.prod(a, axis=axis),
        'div': lambda a: right_reduce(lambda x, y: x / y, a),
        'max': lambda a: np.max(a, axis=axis),
        'min': lambda a: np.min(a, axis=axis),
        'and': lambda a: np.all(a, axis=axis).astype(int),
        'or': lambda a: np.any(a, axis=axis).astype(int),
        # APL symbols (for backwards compatibility)
        '+': lambda a: np.sum(a, axis=axis),
        '-': lambda a: right_reduce(lambda x, y: x - y, a),
        'x': lambda a: np.prod(a, axis=axis),
        '/': lambda a: right_reduce(lambda x, y: x / y, a),
        'max_sym': lambda a: np.max(a, axis=axis),
        'min_sym': lambda a: np.min(a, axis=axis),
    }

    if fn in reductions:
        return reductions[fn](arr)
    return arr


def apl_scan(fn: str, x: ArrayLike, axis: int = -1) -> np.ndarray:
    """Apply scan with given function."""
    arr = ensure_array(x)

    if arr.ndim <= 1:
        axis = -1

    scans = {
        'add': lambda a: np.cumsum(a, axis=axis),
        'mul': lambda a: np.cumprod(a, axis=axis),
        'max': lambda a: np.maximum.accumulate(a, axis=axis),
        'min': lambda a: np.minimum.accumulate(a, axis=axis),
        '+': lambda a: np.cumsum(a, axis=axis),
        'x': lambda a: np.cumprod(a, axis=axis),
    }

    if fn in scans:
        return scans[fn](arr)
    return arr


def apl_compress(mask: ArrayLike, x: ArrayLike) -> np.ndarray:
    """compress dyadic: compress (replicate/boolean selection)"""
    mask = ensure_array(mask)
    arr = ensure_array(x).flatten()

    if np.all((mask == 0) | (mask == 1)):
        # Boolean mask
        return arr[mask.flatten().astype(bool)]
    else:
        # Replicate: each element of mask says how many times to repeat
        result = []
        for m, v in zip(mask.flatten(), arr):
            result.extend([v] * int(m))
        return np.array(result)


def apl_expand(mask: ArrayLike, x: ArrayLike) -> np.ndarray:
    """expand dyadic: expand"""
    mask = ensure_array(mask).flatten()
    arr = ensure_array(x).flatten()

    result = np.zeros(len(mask), dtype=arr.dtype)
    j = 0
    for i, m in enumerate(mask):
        if m:
            if j < len(arr):
                result[i] = arr[j]
                j += 1
    return result


# ============================================================
# Matrix Operations
# ============================================================

def apl_matrix_inverse(x: ArrayLike) -> np.ndarray:
    """matrix_inverse monadic: matrix inverse"""
    arr = ensure_array(x)
    if arr.ndim == 1:
        # For vectors, return reciprocal of sum of squares
        return arr / np.sum(arr ** 2)
    return np.linalg.inv(arr)


def apl_matrix_divide(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """matrix_divide dyadic: matrix divide (solve linear system)"""
    a_arr, b_arr = ensure_array(a), ensure_array(b)
    try:
        return np.linalg.solve(b_arr, a_arr)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(b_arr, a_arr, rcond=None)[0]


def apl_inner_product(fn1: str, fn2: str, a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Inner product: a fn1.fn2 b"""
    a, b = ensure_array(a), ensure_array(b)

    # Special case for +.x (matrix multiply)
    if fn1 in ('add', '+') and fn2 in ('mul', 'x'):
        return np.dot(a, b)

    # General inner product
    if a.ndim == 1 and b.ndim == 1:
        if len(a) != len(b):
            raise ValueError("Vectors must have same length for inner product")
        paired = _apply_dyadic_fn(fn2, a, b)
        return apl_reduce(fn1, paired)

    # Matrix inner product
    return np.dot(a, b)


def apl_outer_product(fn: str, a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Outer product: a o.fn b"""
    a, b = ensure_array(a).flatten(), ensure_array(b).flatten()

    outer_funcs = {
        'add': np.add.outer, '+': np.add.outer,
        'sub': np.subtract.outer, '-': np.subtract.outer,
        'mul': np.multiply.outer, 'x': np.multiply.outer,
        'div': np.divide.outer,
        'mod': np.mod.outer, '|': np.mod.outer,  # residue/modulo
        'max': np.maximum.outer,
        'min': np.minimum.outer,
        'eq': lambda x, y: np.equal.outer(x, y).astype(int), '=': lambda x, y: np.equal.outer(x, y).astype(int),
        'ne': lambda x, y: np.not_equal.outer(x, y).astype(int),
        'lt': lambda x, y: np.less.outer(x, y).astype(int), '<': lambda x, y: np.less.outer(x, y).astype(int),
        'le': lambda x, y: np.less_equal.outer(x, y).astype(int),
        'gt': lambda x, y: np.greater.outer(x, y).astype(int), '>': lambda x, y: np.greater.outer(x, y).astype(int),
        'ge': lambda x, y: np.greater_equal.outer(x, y).astype(int),
        'pow': np.power.outer, '*': np.power.outer,
    }

    if fn in outer_funcs:
        return outer_funcs[fn](a, b)

    return np.outer(a, b)


# ============================================================
# Encode/Decode
# ============================================================

def apl_encode(radix: ArrayLike, value: ArrayLike) -> np.ndarray:
    """encode dyadic: encode (representation in mixed radix)"""
    radix = ensure_array(radix).astype(int).flatten()
    value = int(ensure_array(value).flat[0])

    result = np.zeros(len(radix), dtype=int)
    for i in range(len(radix) - 1, -1, -1):
        if radix[i] == 0:
            result[i] = value
            value = 0
        else:
            result[i] = value % radix[i]
            value //= radix[i]
    return result


def apl_decode(radix: ArrayLike, digits: ArrayLike) -> np.ndarray:
    """decode dyadic: decode (value from mixed radix)"""
    radix = ensure_array(radix).flatten()
    digits = ensure_array(digits).flatten()

    # Handle single radix (polynomial evaluation)
    if len(radix) == 1:
        base = radix[0]
        result = 0
        for d in digits:
            result = result * base + d
        return np.array(result)

    # Mixed radix decode
    result = 0
    multiplier = 1
    for i in range(len(digits) - 1, -1, -1):
        result += digits[i] * multiplier
        if i > 0:
            multiplier *= radix[i]
    return np.array(result)


# ============================================================
# Operators (Higher-order functions)
# ============================================================

def apl_each(fn: Callable, x: ArrayLike) -> np.ndarray:
    """each operator: apply function to each element"""
    arr = ensure_array(x)
    return np.array([fn(item) for item in arr.flat]).reshape(arr.shape)


def apl_each_dyadic(fn: Callable, a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """each operator dyadic: apply function to corresponding pairs"""
    a_arr, b_arr = ensure_array(a), ensure_array(b)
    if a_arr.shape == b_arr.shape:
        return np.array([fn(ai, bi) for ai, bi in zip(a_arr.flat, b_arr.flat)]).reshape(a_arr.shape)
    # Scalar extension
    if a_arr.size == 1:
        return np.array([fn(a_arr.flat[0], bi) for bi in b_arr.flat]).reshape(b_arr.shape)
    if b_arr.size == 1:
        return np.array([fn(ai, b_arr.flat[0]) for ai in a_arr.flat]).reshape(a_arr.shape)
    raise ValueError("Shapes don't match for each operator")


def apl_commute(fn: Callable, a: ArrayLike, b: ArrayLike = None) -> np.ndarray:
    """commute operator: swap arguments or selfie"""
    if b is None:
        # Selfie: f~ x = x f x
        return fn(a, a)
    # Swap: x f~ y = y f x
    return fn(b, a)


def apl_power_operator(fn: Callable, n: int, x: ArrayLike) -> np.ndarray:
    """power operator: apply function n times"""
    result = ensure_array(x)
    for _ in range(n):
        result = fn(result)
    return result


def apl_until(fn: Callable, cond: Callable, x: ArrayLike) -> np.ndarray:
    """power operator with condition: apply until condition met"""
    result = ensure_array(x)
    prev = None
    max_iter = 1000
    for _ in range(max_iter):
        if prev is not None and cond(prev, result):
            break
        prev = result.copy()
        result = fn(result)
    return result


# ============================================================
# Helper Functions
# ============================================================

def _apply_dyadic_fn(fn: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply a dyadic function by name."""
    fn_map = {
        'add': np.add, '+': np.add,
        'sub': np.subtract, '-': np.subtract,
        'mul': np.multiply, 'x': np.multiply,
        'div': np.divide, '/': np.divide,
        'max': np.maximum,
        'min': np.minimum,
        'pow': np.power, '*': np.power,
        'eq': lambda x,y: (x == y).astype(int), '=': lambda x,y: (x == y).astype(int),
        'ne': lambda x,y: (x != y).astype(int),
        'lt': lambda x,y: (x < y).astype(int), '<': lambda x,y: (x < y).astype(int),
        'le': lambda x,y: (x <= y).astype(int),
        'gt': lambda x,y: (x > y).astype(int), '>': lambda x,y: (x > y).astype(int),
        'ge': lambda x,y: (x >= y).astype(int),
    }
    if fn in fn_map:
        return fn_map[fn](a, b)
    return np.multiply(a, b)  # default


# ============================================================
# System Constants
# ============================================================

# Index origin (1-based by default in APL)
IO = 1

# Comparison tolerance
CT = 1e-14

# Empty numeric vector
ZILDE = np.array([], dtype=float)


# ============================================================
# Export all functions
# ============================================================

__all__ = [
    # Utility
    'ensure_array', 'scalar_result',
    # Monadic
    'apl_conjugate', 'apl_negate', 'apl_signum', 'apl_reciprocal',
    'apl_ceiling', 'apl_floor', 'apl_magnitude', 'apl_exponential',
    'apl_natural_log', 'apl_factorial', 'apl_pi_times', 'apl_not',
    'apl_roll', 'apl_where', 'apl_format', 'apl_execute', 'apl_depth',
    'apl_iota', 'apl_shape', 'apl_ravel', 'apl_reverse', 'apl_reverse_first',
    'apl_transpose', 'apl_enclose', 'apl_disclose', 'apl_first',
    'apl_grade_up', 'apl_grade_down', 'apl_unique', 'apl_tally',
    'apl_matrix_inverse', 'apl_left_monadic', 'apl_right_monadic', 'apl_same',
    # Dyadic
    'apl_add', 'apl_subtract', 'apl_multiply', 'apl_divide',
    'apl_maximum', 'apl_minimum', 'apl_power', 'apl_log', 'apl_residue',
    'apl_binomial', 'apl_circular', 'apl_left', 'apl_right',
    'apl_less', 'apl_less_equal', 'apl_equal', 'apl_greater_equal',
    'apl_greater', 'apl_not_equal', 'apl_and', 'apl_or', 'apl_nand', 'apl_nor',
    'apl_index_of', 'apl_reshape', 'apl_catenate', 'apl_catenate_first',
    'apl_rotate', 'apl_rotate_first', 'apl_transpose_dyadic',
    'apl_take', 'apl_drop', 'apl_pick', 'apl_membership', 'apl_find',
    'apl_union', 'apl_intersection', 'apl_without', 'apl_match', 'apl_not_match',
    'apl_index', 'apl_matrix_divide', 'apl_encode', 'apl_decode',
    # Operators
    'apl_reduce', 'apl_scan', 'apl_compress', 'apl_expand',
    'apl_inner_product', 'apl_outer_product',
    'apl_each', 'apl_each_dyadic', 'apl_commute', 'apl_power_operator', 'apl_until',
    # Constants
    'IO', 'CT', 'ZILDE',
]
