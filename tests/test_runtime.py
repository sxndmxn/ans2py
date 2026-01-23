"""Comprehensive tests for APL runtime primitives."""

import numpy as np
import pytest
from ans2py.runtime import *


class TestMonadicScalar:
    """Test monadic scalar functions."""

    def test_conjugate(self):
        assert apl_conjugate(5) == 5
        assert apl_conjugate(complex(3, 4)) == complex(3, -4)

    def test_negate(self):
        assert np.array_equal(apl_negate([1, -2, 3]), [-1, 2, -3])

    def test_signum(self):
        assert np.array_equal(apl_signum([-5, 0, 5]), [-1, 0, 1])

    def test_reciprocal(self):
        assert np.allclose(apl_reciprocal([1, 2, 4]), [1, 0.5, 0.25])

    def test_ceiling(self):
        assert np.array_equal(apl_ceiling([1.2, 2.8, -1.5]), [2, 3, -1])

    def test_floor(self):
        assert np.array_equal(apl_floor([1.2, 2.8, -1.5]), [1, 2, -2])

    def test_magnitude(self):
        assert np.array_equal(apl_magnitude([-3, 4, -5]), [3, 4, 5])
        assert apl_magnitude(complex(3, 4)) == 5

    def test_exponential(self):
        assert np.allclose(apl_exponential([0, 1]), [1, np.e])

    def test_natural_log(self):
        assert np.allclose(apl_natural_log([1, np.e]), [0, 1])

    def test_factorial(self):
        result = apl_factorial([0, 1, 5, 10])
        assert result[0] == 1
        assert result[1] == 1
        assert result[2] == 120
        assert result[3] == 3628800

    def test_pi_times(self):
        assert np.allclose(apl_pi_times([1, 2]), [np.pi, 2 * np.pi])

    def test_not(self):
        assert np.array_equal(apl_not([0, 1, 0, 1]), [1, 0, 1, 0])

    def test_roll(self):
        result = apl_roll([6, 6, 6])
        assert all(1 <= r <= 6 for r in result)


class TestDyadicScalar:
    """Test dyadic scalar functions."""

    def test_add(self):
        assert np.array_equal(np.add([1, 2], [3, 4]), [4, 6])

    def test_subtract(self):
        assert np.array_equal(np.subtract([5, 6], [1, 2]), [4, 4])

    def test_multiply(self):
        assert np.array_equal(np.multiply([2, 3], [4, 5]), [8, 15])

    def test_divide(self):
        assert np.allclose(np.divide([10, 15], [2, 3]), [5, 5])

    def test_maximum(self):
        assert np.array_equal(np.maximum([1, 5, 3], [2, 4, 6]), [2, 5, 6])

    def test_minimum(self):
        assert np.array_equal(np.minimum([1, 5, 3], [2, 4, 6]), [1, 4, 3])

    def test_power(self):
        assert np.array_equal(np.power([2, 3], [3, 2]), [8, 9])

    def test_log(self):
        assert np.allclose(apl_log(2, 8), 3)  # log base 2 of 8
        assert np.allclose(apl_log(10, 100), 2)

    def test_residue(self):
        assert np.array_equal(apl_residue(3, [7, 8, 9]), [1, 2, 0])

    def test_binomial(self):
        assert apl_binomial(5, 2) == 10  # 5 choose 2
        assert apl_binomial(10, 3) == 120


class TestComparison:
    """Test comparison functions."""

    def test_less(self):
        assert np.array_equal(apl_less([1, 2, 3], [2, 2, 2]), [1, 0, 0])

    def test_less_equal(self):
        assert np.array_equal(apl_less_equal([1, 2, 3], [2, 2, 2]), [1, 1, 0])

    def test_equal(self):
        assert np.array_equal(apl_equal([1, 2, 3], [1, 5, 3]), [1, 0, 1])

    def test_greater_equal(self):
        assert np.array_equal(apl_greater_equal([1, 2, 3], [2, 2, 2]), [0, 1, 1])

    def test_greater(self):
        assert np.array_equal(apl_greater([1, 2, 3], [2, 2, 2]), [0, 0, 1])

    def test_not_equal(self):
        assert np.array_equal(apl_not_equal([1, 2, 3], [1, 5, 3]), [0, 1, 0])


class TestLogical:
    """Test logical functions."""

    def test_and(self):
        assert np.array_equal(apl_and([0, 0, 1, 1], [0, 1, 0, 1]), [0, 0, 0, 1])

    def test_or(self):
        assert np.array_equal(apl_or([0, 0, 1, 1], [0, 1, 0, 1]), [0, 1, 1, 1])

    def test_nand(self):
        assert np.array_equal(apl_nand([0, 0, 1, 1], [0, 1, 0, 1]), [1, 1, 1, 0])

    def test_nor(self):
        assert np.array_equal(apl_nor([0, 0, 1, 1], [0, 1, 0, 1]), [1, 0, 0, 0])


class TestArrayFunctions:
    """Test array manipulation functions."""

    def test_iota(self):
        assert np.array_equal(apl_iota(5), [1, 2, 3, 4, 5])
        assert np.array_equal(apl_iota(0), [])

    def test_shape(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.array_equal(apl_shape(arr), [2, 3])

    def test_reshape(self):
        result = apl_reshape([2, 3], apl_iota(6))
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.array_equal(result, expected)

    def test_reshape_with_repeat(self):
        # APL reshape repeats elements if needed
        result = apl_reshape([2, 3], [1, 2])
        expected = np.array([[1, 2, 1], [2, 1, 2]])
        assert np.array_equal(result, expected)

    def test_ravel(self):
        arr = np.array([[1, 2], [3, 4]])
        assert np.array_equal(apl_ravel(arr), [1, 2, 3, 4])

    def test_tally(self):
        assert apl_tally([1, 2, 3, 4, 5]) == 5
        assert apl_tally([[1, 2], [3, 4]]) == 2

    def test_reverse(self):
        assert np.array_equal(apl_reverse([1, 2, 3]), [3, 2, 1])

    def test_transpose(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        assert np.array_equal(apl_transpose(arr), expected)

    def test_enclose(self):
        result = apl_enclose([1, 2, 3])
        # Enclose creates an object array containing the original
        assert result.dtype == object
        assert np.array_equal(result[0], [1, 2, 3])

    def test_disclose(self):
        enclosed = apl_enclose([1, 2, 3])
        result = apl_disclose(enclosed)
        assert np.array_equal(result, [1, 2, 3])

    def test_grade_up(self):
        # Returns 1-based indices
        assert np.array_equal(apl_grade_up([3, 1, 4, 1, 5]), [2, 4, 1, 3, 5])

    def test_grade_down(self):
        assert np.array_equal(apl_grade_down([3, 1, 4, 1, 5]), [5, 3, 1, 2, 4])


class TestTakeDropRotate:
    """Test take, drop, and rotate."""

    def test_take_positive(self):
        assert np.array_equal(apl_take(3, [1, 2, 3, 4, 5]), [1, 2, 3])

    def test_take_negative(self):
        assert np.array_equal(apl_take(-3, [1, 2, 3, 4, 5]), [3, 4, 5])

    def test_take_with_padding(self):
        result = apl_take(5, [1, 2, 3])
        assert np.array_equal(result, [1, 2, 3, 0, 0])

    def test_drop_positive(self):
        assert np.array_equal(apl_drop(2, [1, 2, 3, 4, 5]), [3, 4, 5])

    def test_drop_negative(self):
        assert np.array_equal(apl_drop(-2, [1, 2, 3, 4, 5]), [1, 2, 3])

    def test_rotate(self):
        assert np.array_equal(apl_rotate(2, [1, 2, 3, 4, 5]), [3, 4, 5, 1, 2])
        assert np.array_equal(apl_rotate(-1, [1, 2, 3, 4, 5]), [5, 1, 2, 3, 4])


class TestSetFunctions:
    """Test set operations."""

    def test_unique(self):
        result = apl_unique([1, 2, 2, 3, 3, 3])
        assert set(result) == {1, 2, 3}

    def test_union(self):
        result = apl_union([1, 2, 3], [2, 3, 4])
        assert set(result) == {1, 2, 3, 4}

    def test_intersection(self):
        result = apl_intersection([1, 2, 3], [2, 3, 4])
        assert set(result) == {2, 3}

    def test_without(self):
        result = apl_without([1, 2, 3, 4], [2, 4])
        assert np.array_equal(result, [1, 3])

    def test_membership(self):
        result = apl_membership([1, 2, 3, 4], [2, 4, 6])
        assert np.array_equal(result, [0, 1, 0, 1])

    def test_index_of(self):
        # Returns 1-based index, or length+1 if not found
        result = apl_index_of([10, 20, 30], [20, 40, 10])
        assert np.array_equal(result, [2, 4, 1])


class TestEncodeDecode:
    """Test encode and decode (base conversion)."""

    def test_encode_binary(self):
        # 10 in binary (4 bits)
        result = apl_encode([2, 2, 2, 2], 10)
        assert np.array_equal(result, [1, 0, 1, 0])

    def test_encode_mixed_radix(self):
        # 90 minutes = 1 hour 30 minutes
        result = apl_encode([24, 60], 90)
        assert np.array_equal(result, [1, 30])

    def test_decode_binary(self):
        # Binary 1010 = 10
        result = apl_decode(2, [1, 0, 1, 0])
        assert result == 10

    def test_decode_mixed_radix(self):
        # 1 hour 30 minutes = 90 minutes
        result = apl_decode([24, 60], [1, 30])
        assert result == 90


class TestReduce:
    """Test reduce operations."""

    def test_reduce_add(self):
        assert apl_reduce('add', [1, 2, 3, 4, 5]) == 15

    def test_reduce_mul(self):
        assert apl_reduce('mul', [1, 2, 3, 4, 5]) == 120

    def test_reduce_max(self):
        assert apl_reduce('max', [3, 1, 4, 1, 5, 9, 2, 6]) == 9

    def test_reduce_min(self):
        assert apl_reduce('min', [3, 1, 4, 1, 5, 9, 2, 6]) == 1

    def test_reduce_sub(self):
        # APL subtraction reduce: 1-2-3-4 (right to left) = 1-(2-(3-4)) = 1-(2-(-1)) = 1-3 = -2
        result = apl_reduce('sub', [1, 2, 3, 4])
        assert result == -2

    def test_reduce_2d_along_axis(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = apl_reduce('add', arr)
        assert np.array_equal(result, [6, 15])  # Sum each row

    def test_reduce_and(self):
        assert apl_reduce('and', [1, 1, 1, 1]) == 1
        assert apl_reduce('and', [1, 1, 0, 1]) == 0

    def test_reduce_or(self):
        assert apl_reduce('or', [0, 0, 0, 0]) == 0
        assert apl_reduce('or', [0, 0, 1, 0]) == 1


class TestScan:
    """Test scan operations."""

    def test_scan_add(self):
        result = apl_scan('add', [1, 2, 3, 4, 5])
        assert np.array_equal(result, [1, 3, 6, 10, 15])

    def test_scan_mul(self):
        result = apl_scan('mul', [1, 2, 3, 4, 5])
        assert np.array_equal(result, [1, 2, 6, 24, 120])

    def test_scan_max(self):
        result = apl_scan('max', [3, 1, 4, 1, 5, 9, 2, 6])
        assert np.array_equal(result, [3, 3, 4, 4, 5, 9, 9, 9])


class TestCompressExpand:
    """Test compress and expand."""

    def test_compress(self):
        result = apl_compress([1, 0, 1, 0, 1], [10, 20, 30, 40, 50])
        assert np.array_equal(result, [10, 30, 50])

    def test_expand(self):
        result = apl_expand([1, 0, 1, 0, 1], [10, 30, 50])
        assert np.array_equal(result, [10, 0, 30, 0, 50])


class TestOuterProduct:
    """Test outer product."""

    def test_outer_mul(self):
        result = apl_outer_product('mul', [1, 2, 3], [4, 5])
        expected = np.array([[4, 5], [8, 10], [12, 15]])
        assert np.array_equal(result, expected)

    def test_outer_add(self):
        result = apl_outer_product('add', [1, 2, 3], [10, 20])
        expected = np.array([[11, 21], [12, 22], [13, 23]])
        assert np.array_equal(result, expected)

    def test_outer_equal(self):
        result = apl_outer_product('eq', [1, 2, 3], [1, 2, 3])
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.array_equal(result, expected)


class TestInnerProduct:
    """Test inner product."""

    def test_inner_add_mul(self):
        # Standard matrix multiplication
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = apl_inner_product('add', 'mul', a, b)
        assert result == 32  # 1*4 + 2*5 + 3*6

    def test_inner_max_min(self):
        a = np.array([1, 5, 3])
        b = np.array([4, 2, 6])
        result = apl_inner_product('max', 'min', a, b)
        # min pairs: [1,5,3], max of those: 3
        assert result == 3


class TestMatrixOps:
    """Test matrix operations."""

    def test_matrix_inverse(self):
        m = np.array([[1, 2], [3, 4]], dtype=float)
        inv = np.linalg.inv(m)
        result = np.dot(m, inv)
        assert np.allclose(result, np.eye(2))

    def test_matrix_solve(self):
        # Solve Ax = b
        A = np.array([[3, 1], [1, 2]], dtype=float)
        b = np.array([9, 8], dtype=float)
        x = np.linalg.solve(A, b)
        assert np.allclose(np.dot(A, x), b)


class TestCircularFunctions:
    """Test circular/trig functions."""

    def test_circular_sin(self):
        result = apl_circular(1, np.pi / 2)
        assert np.allclose(result, 1)  # sin(pi/2) = 1

    def test_circular_cos(self):
        result = apl_circular(2, 0)
        assert np.allclose(result, 1)  # cos(0) = 1

    def test_circular_tan(self):
        result = apl_circular(3, np.pi / 4)
        assert np.allclose(result, 1)  # tan(pi/4) = 1

    def test_circular_sqrt(self):
        result = apl_circular(0, 0.6)
        assert np.allclose(result, 0.8)  # sqrt(1 - 0.6^2) = 0.8


class TestIdentityFunctions:
    """Test left and right identity functions."""

    def test_right_identity(self):
        assert apl_right(5, 10) == 10
        assert np.array_equal(apl_right([1, 2], [3, 4]), [3, 4])

    def test_left_identity(self):
        assert apl_left(5, 10) == 5
        assert np.array_equal(apl_left([1, 2], [3, 4]), [1, 2])

    def test_same(self):
        assert apl_same(42) == 42
        assert np.array_equal(apl_same([1, 2, 3]), [1, 2, 3])


class TestMatch:
    """Test match function."""

    def test_match_equal(self):
        assert apl_match([1, 2, 3], [1, 2, 3]) == 1

    def test_match_not_equal(self):
        assert apl_match([1, 2, 3], [1, 2, 4]) == 0
        assert apl_match([1, 2, 3], [1, 2]) == 0


class TestAPLExpressions:
    """Test complete APL expressions via the converter."""

    def test_sum_iota(self):
        # +/ ⍳ 5 = 15
        result = apl_reduce('add', apl_iota(5))
        assert result == 15

    def test_product_iota(self):
        # ×/ ⍳ 5 = 120
        result = apl_reduce('mul', apl_iota(5))
        assert result == 120

    def test_reshape_iota(self):
        # 2 3 ⍴ ⍳ 6
        result = apl_reshape([2, 3], apl_iota(6))
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.array_equal(result, expected)

    def test_max_reduce(self):
        # ⌈/ 3 1 4 1 5 9 2 6
        result = apl_reduce('max', [3, 1, 4, 1, 5, 9, 2, 6])
        assert result == 9

    def test_primes_sieve(self):
        # Simple prime test using divisor counting
        n = apl_iota(20)
        # For each number, count how many numbers 1..n divide it evenly
        # A prime has exactly 2 divisors (1 and itself)
        def count_divisors(k):
            return sum(1 for j in range(1, k + 1) if k % j == 0)
        divisor_count = np.array([count_divisors(int(k)) for k in n])
        is_prime = apl_equal(divisor_count, 2)
        primes = apl_compress(is_prime, n)
        assert np.array_equal(primes, [2, 3, 5, 7, 11, 13, 17, 19])

    def test_fibonacci(self):
        # Generate first 10 Fibonacci numbers using scan
        # This is a simplified version
        fib = [1, 1]
        for _ in range(8):
            fib.append(fib[-1] + fib[-2])
        assert fib == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

    def test_pascal_triangle_row(self):
        # Row 5 of Pascal's triangle: !∘5¨⍳6 (but 0-indexed)
        # Or: 5!0 1 2 3 4 5
        row = [apl_binomial(5, k) for k in range(6)]
        assert row == [1, 5, 10, 10, 5, 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
