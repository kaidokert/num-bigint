// FixedWidthBigUint: const-num-traits + CiosRowOps for num-bigint.
//
// Design: stores `inner: BigUint` directly and delegates all arithmetic to it.
// No padding, no Vec roundtrips. `n_limbs` carries the declared width for
// the three operations that genuinely need it:
//   - bits_precision()          → ring width, not value width
//   - CiosRowOps::word_count()  → stable limb count for the inner loop
//   - OverflowingAdd            → overflow detection at declared width
//
// WrappingSub is the one place where BigUint's unsigned arithmetic would panic
// (lhs < rhs); we handle that case by adding 2^width.
//
// Everything else — Add/Sub/Mul/Div/Rem/Wrapping*/Checked*/Carrying* — is
// pure delegation. BigUint grows to hold any result; that's correct and
// cheaper than the previous Vec-roundtrip approach.

use super::IntDigits;
use crate::big_digit::{self, BigDigit};
use crate::std_alloc::Vec;
use crate::BigInt;
use crate::BigUint;
use const_num_traits::{
    ops::{
        bits::{BitsPrecision, WithPrecision},
        byte_slice::ByteSliceError,
        carrying::CarryingMul,
        checked::{CheckedAdd, CheckedMul},
        overflowing::{OverflowingAdd, OverflowingMul, OverflowingSub},
        wrapping::{WrappingAdd, WrappingMul, WrappingSub},
    },
    BorrowingSub, FromByteSlice, HasPersonality, Nct, One, Parity, ToBytes, Zero,
};
use modmath_cios::CiosRowOps;
use num_traits::Zero as NtZero;
use core::cmp::Ordering;
use core::ops::{
    Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, RemAssign, Shl, Shr, ShrAssign, Sub,
};

const DIGIT_BITS: u32 = big_digit::BITS as u32;
const DIGIT_BYTES: usize = (DIGIT_BITS / 8) as usize;

// ── FixedWidthBigUint ─────────────────────────────────────────────────────────

/// A [`BigUint`] with a declared stable limb count.
///
/// `inner` is a plain, normalized `BigUint`; `n_limbs` is the declared width.
/// Arithmetic is delegated directly to `BigUint`. Only `CiosRowOps`,
/// `bits_precision`, and `OverflowingAdd` (overflow detection) require the
/// declared `n_limbs`.
pub struct FixedWidthBigUint {
    inner: BigUint,
    n_limbs: usize,
}

impl FixedWidthBigUint {
    /// Construct from a `BigUint` with a declared width.
    pub fn new(value: BigUint, n_limbs: usize) -> Self {
        Self { inner: value, n_limbs }
    }

    /// Construct from a `u64` literal.
    pub fn from_u64(value: u64, n_limbs: usize) -> Self {
        Self { inner: BigUint::from(value), n_limbs }
    }

    #[inline]
    pub fn n_limbs(&self) -> usize {
        self.n_limbs
    }

    #[inline]
    pub fn as_biguint(&self) -> &BigUint {
        &self.inner
    }

    pub fn into_biguint(self) -> BigUint {
        self.inner
    }

    /// 2^(n_limbs * DIGIT_BITS): the modular ceiling used by WrappingSub and
    /// OverflowingAdd.
    fn width_modulus(n: usize) -> BigUint {
        BigUint::from(1u32) << (n * DIGIT_BITS as usize)
    }
}

// ── Clone / Debug / Hash ──────────────────────────────────────────────────────

impl Clone for FixedWidthBigUint {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), n_limbs: self.n_limbs }
    }
}

impl core::fmt::Debug for FixedWidthBigUint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FixedWidthBigUint({:?}, n={})", self.inner, self.n_limbs)
    }
}

impl core::hash::Hash for FixedWidthBigUint {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
        self.n_limbs.hash(state);
    }
}

// ── Equality / Ordering ───────────────────────────────────────────────────────

impl PartialEq for FixedWidthBigUint {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl Eq for FixedWidthBigUint {}

impl Ord for FixedWidthBigUint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}
impl PartialOrd for FixedWidthBigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Default for FixedWidthBigUint {
    fn default() -> Self {
        Self { inner: BigUint::from(0u32), n_limbs: 1 }
    }
}

// ── From<uN> — minimal-width policy ──────────────────────────────────────────

macro_rules! impl_from_uint {
    ($($t:ty),*) => {$(
        impl From<$t> for FixedWidthBigUint {
            fn from(v: $t) -> Self {
                let inner = BigUint::from(v);
                let n = inner.data.len().max(1);
                Self { inner, n_limbs: n }
            }
        }
    )*};
}
impl_from_uint!(u8, u16, u32, u64, u128, usize);

// ── Personality ───────────────────────────────────────────────────────────────

impl HasPersonality for FixedWidthBigUint {
    type P = Nct;
}

// ── Identity values ───────────────────────────────────────────────────────────

impl Zero for FixedWidthBigUint {
    fn zero() -> Self {
        Self::default()
    }
    fn is_zero(&self) -> bool {
        NtZero::is_zero(&self.inner)
    }
    fn set_zero(&mut self) {
        self.inner = BigUint::from(0u32);
    }
}

impl One for FixedWidthBigUint {
    fn one() -> Self {
        Self { inner: BigUint::from(1u32), n_limbs: 1 }
    }
    fn is_one(&self) -> bool {
        self.inner.data.as_slice() == &[1 as BigDigit]
    }
    fn set_one(&mut self) {
        self.inner = BigUint::from(1u32);
    }
}

// ── BitsPrecision / WithPrecision ─────────────────────────────────────────────

impl BitsPrecision for FixedWidthBigUint {
    fn bits_precision(&self) -> u32 {
        (self.n_limbs * DIGIT_BITS as usize) as u32
    }
}

impl BitsPrecision for &FixedWidthBigUint {
    fn bits_precision(&self) -> u32 {
        (self.n_limbs * DIGIT_BITS as usize) as u32
    }
}

impl WithPrecision for FixedWidthBigUint {
    fn widen_to_precision(self, _bits_precision: u32) -> Self {
        self
    }

    fn zero_with_precision(bits_precision: u32) -> Self where Self: Zero {
        let n = bits_precision.div_ceil(DIGIT_BITS) as usize;
        Self { inner: BigUint::from(0u32), n_limbs: n }
    }

    fn one_with_precision(bits_precision: u32) -> Self where Self: One {
        let n = bits_precision.div_ceil(DIGIT_BITS) as usize;
        Self { inner: BigUint::from(1u32), n_limbs: n }
    }

    fn widen_to_precision_of(self, witness: &Self) -> Self {
        Self { inner: self.inner, n_limbs: witness.n_limbs }
    }

    fn zero_with_precision_of(witness: &Self) -> Self where Self: Zero {
        Self { inner: BigUint::from(0u32), n_limbs: witness.n_limbs }
    }

    fn one_with_precision_of(witness: &Self) -> Self where Self: One {
        Self { inner: BigUint::from(1u32), n_limbs: witness.n_limbs }
    }
}

// ── Parity ────────────────────────────────────────────────────────────────────

impl Parity for FixedWidthBigUint {
    fn is_odd(self) -> bool {
        self.inner.data.first().map_or(false, |d| d & 1 == 1)
    }
    fn is_even(self) -> bool { !Parity::is_odd(self) }
}

impl Parity for &FixedWidthBigUint {
    fn is_odd(self) -> bool {
        self.inner.data.first().map_or(false, |d| d & 1 == 1)
    }
    fn is_even(self) -> bool { !Parity::is_odd(self) }
}

// ── Operator traits ───────────────────────────────────────────────────────────

macro_rules! fw_binop {
    ($Trait:ident, $method:ident, $op:tt) => {
        impl $Trait for FixedWidthBigUint {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                let n = self.n_limbs.max(rhs.n_limbs);
                Self { inner: self.inner $op rhs.inner, n_limbs: n }
            }
        }
        impl $Trait<&FixedWidthBigUint> for FixedWidthBigUint {
            type Output = FixedWidthBigUint;
            fn $method(self, rhs: &FixedWidthBigUint) -> FixedWidthBigUint {
                let n = self.n_limbs.max(rhs.n_limbs);
                Self { inner: self.inner $op rhs.inner.clone(), n_limbs: n }
            }
        }
        impl $Trait<FixedWidthBigUint> for &FixedWidthBigUint {
            type Output = FixedWidthBigUint;
            fn $method(self, rhs: FixedWidthBigUint) -> FixedWidthBigUint {
                let n = self.n_limbs.max(rhs.n_limbs);
                FixedWidthBigUint { inner: self.inner.clone() $op rhs.inner, n_limbs: n }
            }
        }
        impl $Trait<&FixedWidthBigUint> for &FixedWidthBigUint {
            type Output = FixedWidthBigUint;
            fn $method(self, rhs: &FixedWidthBigUint) -> FixedWidthBigUint {
                let n = self.n_limbs.max(rhs.n_limbs);
                FixedWidthBigUint { inner: self.inner.clone() $op rhs.inner.clone(), n_limbs: n }
            }
        }
    };
}

fw_binop!(Add, add, +);
fw_binop!(Sub, sub, -);
fw_binop!(Mul, mul, *);
fw_binop!(Div, div, /);
fw_binop!(Rem, rem, %);
fw_binop!(BitAnd, bitand, &);
fw_binop!(BitOr, bitor, |);
fw_binop!(BitXor, bitxor, ^);

impl RemAssign for FixedWidthBigUint {
    fn rem_assign(&mut self, rhs: Self) {
        self.inner = self.inner.clone() % rhs.inner;
    }
}
impl RemAssign<&FixedWidthBigUint> for FixedWidthBigUint {
    fn rem_assign(&mut self, rhs: &Self) {
        self.inner = self.inner.clone() % rhs.inner.clone();
    }
}

impl Shr<usize> for FixedWidthBigUint {
    type Output = Self;
    fn shr(self, n: usize) -> Self {
        let limbs = self.n_limbs;
        Self { inner: self.inner >> n, n_limbs: limbs }
    }
}
impl ShrAssign<usize> for FixedWidthBigUint {
    fn shr_assign(&mut self, n: usize) {
        self.inner >>= n;
    }
}
impl Shl<usize> for FixedWidthBigUint {
    type Output = Self;
    fn shl(self, n: usize) -> Self {
        let limbs = self.n_limbs;
        Self { inner: self.inner << n, n_limbs: limbs }
    }
}

// ── Wrapping arithmetic ───────────────────────────────────────────────────────
// WrappingAdd/Mul: BigUint never overflows — natural arithmetic is correct.
// WrappingSub: BigUint panics on lhs < rhs; fold in the 2^width addend instead.

impl WrappingAdd for FixedWidthBigUint {
    type Output = Self;
    fn wrapping_add(self, rhs: Self) -> Self {
        let n = self.n_limbs.max(rhs.n_limbs);
        let sum = self.inner + rhs.inner;
        let width = n * DIGIT_BITS as usize;
        // Mask to declared width, consistent with wrapping_sub.
        // EEA sums stay < modulus ≤ 2^width so this is a no-op on the inv path;
        // it correctly removes the 2^width that wrapping_sub injects on underflow.
        let inner = if sum.bits() > width as u64 {
            sum & (Self::width_modulus(n) - BigUint::from(1u32))
        } else {
            sum
        };
        Self { inner, n_limbs: n }
    }
}

impl WrappingSub for FixedWidthBigUint {
    type Output = Self;
    fn wrapping_sub(self, rhs: Self) -> Self {
        let n = self.n_limbs.max(rhs.n_limbs);
        let inner = if self.inner >= rhs.inner {
            self.inner - rhs.inner
        } else {
            Self::width_modulus(n) + self.inner - rhs.inner
        };
        Self { inner, n_limbs: n }
    }
}

impl WrappingMul for FixedWidthBigUint {
    type Output = Self;
    fn wrapping_mul(self, rhs: Self) -> Self {
        let n = self.n_limbs.max(rhs.n_limbs);
        Self { inner: self.inner * rhs.inner, n_limbs: n }
    }
}

impl WrappingAdd for &FixedWidthBigUint {
    type Output = FixedWidthBigUint;
    fn wrapping_add(self, rhs: Self) -> FixedWidthBigUint {
        self.clone().wrapping_add(rhs.clone())
    }
}
impl WrappingSub for &FixedWidthBigUint {
    type Output = FixedWidthBigUint;
    fn wrapping_sub(self, rhs: Self) -> FixedWidthBigUint {
        self.clone().wrapping_sub(rhs.clone())
    }
}

// ── Overflowing arithmetic ────────────────────────────────────────────────────

impl OverflowingAdd for FixedWidthBigUint {
    type Output = Self;
    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        // Heap-backed: addition grows to fit, never truncates. Flag always false.
        (self.wrapping_add(rhs), false)
    }
}

impl OverflowingSub for FixedWidthBigUint {
    type Output = Self;
    fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let n = self.n_limbs.max(rhs.n_limbs);
        if self.inner >= rhs.inner {
            (Self { inner: self.inner - rhs.inner, n_limbs: n }, false)
        } else {
            let inner = Self::width_modulus(n) + self.inner - rhs.inner;
            (Self { inner, n_limbs: n }, true)
        }
    }
}

// ── Checked arithmetic ────────────────────────────────────────────────────────

impl OverflowingMul for FixedWidthBigUint {
    type Output = Self;
    fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        // FixedWidthBigUint is heap-backed: the product grows to fit, never
        // truncates. Overflow never occurs; the flag is always false.
        // Consistent with wrapping_mul and checked_mul on this carrier.
        (self.wrapping_mul(rhs), false)
    }
}

impl CheckedAdd for FixedWidthBigUint {
    type Output = Self;
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let n = self.n_limbs.max(rhs.n_limbs);
        Some(Self { inner: self.inner + rhs.inner, n_limbs: n })
    }
}

impl CheckedMul for FixedWidthBigUint {
    type Output = Self;
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        let n = self.n_limbs.max(rhs.n_limbs);
        Some(Self { inner: self.inner * rhs.inner, n_limbs: n })
    }
}

// ── BorrowingSub ──────────────────────────────────────────────────────────────

impl BorrowingSub for FixedWidthBigUint {
    type Output = Self;
    fn borrowing_sub(self, rhs: Self, borrow: bool) -> (Self, bool) {
        let n = self.n_limbs.max(rhs.n_limbs);
        let lhs = BigInt::from(self.inner);
        let rhs_int = BigInt::from(rhs.inner) + BigInt::from(borrow as u32);
        let diff = lhs - rhs_int;
        let (inner, borrow_out) = if diff < BigInt::from(0i32) {
            let wrapped = (diff + BigInt::from(Self::width_modulus(n)))
                .to_biguint()
                .expect("borrowing_sub: wrapped value must be non-negative");
            (wrapped, true)
        } else {
            (diff.to_biguint().expect("borrowing_sub: non-negative"), false)
        };
        (Self { inner, n_limbs: n }, borrow_out)
    }
}

// ── CarryingMul ───────────────────────────────────────────────────────────────

impl CarryingMul for FixedWidthBigUint {
    type Unsigned = Self;
    type Output = Self;

    fn carrying_mul(self, rhs: Self, carry: Self) -> (Self, Self) {
        let n = self.n_limbs.max(rhs.n_limbs);
        let width = n * DIGIT_BITS as usize;
        let product = self.inner * rhs.inner + carry.inner;
        let modulus = Self::width_modulus(n);
        let lo = product.clone() & (&modulus - BigUint::from(1u32));
        let hi = product >> width;
        (Self { inner: lo, n_limbs: n }, Self { inner: hi, n_limbs: n })
    }

    fn carrying_mul_add(self, rhs: Self, carry: Self, add: Self) -> (Self, Self) {
        let n = self.n_limbs.max(rhs.n_limbs);
        let width = n * DIGIT_BITS as usize;
        let product = self.inner * rhs.inner + carry.inner + add.inner;
        let modulus = Self::width_modulus(n);
        let lo = product.clone() & (&modulus - BigUint::from(1u32));
        let hi = product >> width;
        (Self { inner: lo, n_limbs: n }, Self { inner: hi, n_limbs: n })
    }
}

// ── ToBytes / FromByteSlice ───────────────────────────────────────────────────

impl ToBytes for FixedWidthBigUint {
    type Bytes = Vec<u8>;
    fn to_be_bytes(self) -> Vec<u8> {
        let expected = self.n_limbs * DIGIT_BYTES;
        let bytes = self.inner.to_bytes_be();
        if bytes.len() >= expected {
            bytes
        } else {
            let mut padded = vec![0u8; expected - bytes.len()];
            padded.extend_from_slice(&bytes);
            padded
        }
    }
    fn to_le_bytes(self) -> Vec<u8> {
        let expected = self.n_limbs * DIGIT_BYTES;
        let mut bytes = self.inner.to_bytes_le();
        bytes.resize(expected, 0);
        bytes
    }
}

impl FromByteSlice for FixedWidthBigUint {
    fn from_be_slice(bytes: &[u8]) -> Result<Self, ByteSliceError> {
        let n = bytes.len().div_ceil(DIGIT_BYTES).max(1);
        Ok(Self { inner: BigUint::from_bytes_be(bytes), n_limbs: n })
    }
    fn from_le_slice(bytes: &[u8]) -> Result<Self, ByteSliceError> {
        let n = bytes.len().div_ceil(DIGIT_BYTES).max(1);
        Ok(Self { inner: BigUint::from_bytes_le(bytes), n_limbs: n })
    }
}

// ── CiosRowOps ────────────────────────────────────────────────────────────────
//
// word_count() → declared n_limbs (never changes).
// word(i)      → inner.data[i] or 0; normalization means data may be shorter
//                than n_limbs, so unwrap_or(0) is correct.
//
// mul_acc_row / mul_acc_shift_row write directly into acc.inner.data.
// Before writing: ensure data has n_limbs entries (may temporarily violate
// BigUint's no-trailing-zero invariant). After writing: restore the invariant
// via IntDigits::normalize so BigUint comparisons are correct for subsequent ops.

impl CiosRowOps for FixedWidthBigUint {
    type Word = crate::Digit;

    #[inline]
    fn word_count(&self) -> usize {
        self.n_limbs
    }

    #[inline]
    fn word(&self, i: usize) -> crate::Digit {
        self.inner.data.get(i).copied().unwrap_or(0)
    }

    fn mul_acc_row(
        scalar: crate::Digit,
        multiplicand: &Self,
        acc: &mut Self,
        carry_in: crate::Digit,
    ) -> crate::Digit {
        let n = multiplicand.n_limbs;
        if acc.inner.data.len() < n {
            acc.inner.data.resize(n, 0);
        }
        let s = scalar as u128;
        let mut carry = carry_in as u128;
        let mut j = 0;
        while j < n {
            let m = multiplicand.inner.data.get(j).copied().unwrap_or(0) as u128;
            let a = acc.inner.data.get(j).copied().unwrap_or(0) as u128;
            let product = s * m + a + carry;
            acc.inner.data[j] = product as crate::Digit;
            carry = product >> DIGIT_BITS;
            j += 1;
        }
        IntDigits::normalize(&mut acc.inner);
        carry as crate::Digit
    }

    fn mul_acc_shift_row(
        scalar: crate::Digit,
        multiplicand: &Self,
        acc: &mut Self,
        acc_hi: crate::Digit,
    ) -> crate::Digit {
        let n = multiplicand.n_limbs;
        if acc.inner.data.len() < n {
            acc.inner.data.resize(n, 0);
        }
        let s = scalar as u128;
        // Word 0: discard the shifted-out low digit, keep carry.
        let p0 = s * multiplicand.inner.data.get(0).copied().unwrap_or(0) as u128
            + acc.inner.data.get(0).copied().unwrap_or(0) as u128;
        let mut carry = (p0 >> DIGIT_BITS) as u64;
        // Words 1..n: accumulate and shift down one position.
        let mut j = 1;
        while j < n {
            let m = multiplicand.inner.data.get(j).copied().unwrap_or(0) as u128;
            let a = acc.inner.data.get(j).copied().unwrap_or(0) as u128;
            let product = s * m + a + carry as u128;
            acc.inner.data[j - 1] = product as crate::Digit;
            carry = (product >> DIGIT_BITS) as u64;
            j += 1;
        }
        let (sum, overflow) = acc_hi.overflowing_add(carry);
        if n > 0 {
            acc.inner.data[n - 1] = sum;
        }
        IntDigits::normalize(&mut acc.inner);
        overflow as crate::Digit
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use const_num_traits::ops::bits::BitsPrecision;
    use const_num_traits::{One, WithPrecision, Zero};

    fn fw(value: u64, n: usize) -> FixedWidthBigUint {
        FixedWidthBigUint::from_u64(value, n)
    }

    #[test]
    fn zero_plus_x_equals_x() {
        // The EEA clone idiom: zero(0) + x(k) must equal x(k).
        let zero = FixedWidthBigUint::zero();
        let x = fw(12345, 4);
        let result = zero + x.clone();
        assert_eq!(result, x);
        assert_eq!(result.n_limbs(), 4);
    }

    #[test]
    fn word_count_stable_after_sub() {
        let a = fw(100, 2);
        let b = fw(99, 2);
        let c = a - b;
        assert_eq!(c.n_limbs(), 2);
        assert_eq!(c.word_count(), 2);
        assert_eq!(c.word(0), 1);
        assert_eq!(c.word(1), 0);
    }

    #[test]
    fn bits_precision() {
        assert_eq!(fw(0, 4).bits_precision(), 4 * DIGIT_BITS);
    }

    #[test]
    fn wrapping_sub_underflow() {
        let zero = fw(0, 1);
        let one = fw(1, 1);
        let result = zero.wrapping_sub(one);
        assert_eq!(result.word(0), crate::Digit::MAX);
    }

    #[test]
    fn wrapping_add_masks_consistent_with_sub() {
        // wrapping_add masks to declared width, same as wrapping_sub.
        // This ensures diff.wrapping_add(m) corrects the 2^width that
        // wrapping_sub(a, b) injects when a < b.
        let max = FixedWidthBigUint {
            inner: (BigUint::from(1u32) << DIGIT_BITS as usize) - BigUint::from(1u32),
            n_limbs: 1,
        };
        let one = fw(1, 1);
        // MAX + 1 wraps to 0
        let result = max.clone().wrapping_add(one.clone());
        assert_eq!(result.word(0), 0);
        // overflowing_add delegates to wrapping_add; flag always false on heap carrier
        let (r2, flag) = max.overflowing_add(one);
        assert!(!flag);
        assert_eq!(r2.word(0), 0);
    }

    #[test]
    fn overflowing_sub_detects_underflow() {
        let (_, borrow) = fw(3, 1).overflowing_sub(fw(10, 1));
        assert!(borrow);
        let (diff, no_borrow) = fw(10, 1).overflowing_sub(fw(3, 1));
        assert!(!no_borrow);
        assert_eq!(diff.word(0), 7);
    }

    #[test]
    fn carrying_mul_splits() {
        let a = fw(u32::MAX as u64, 1);
        let b = fw(u32::MAX as u64, 1);
        let (lo, hi) = a.carrying_mul(b, fw(0, 1));
        let expected: u64 = (u32::MAX as u64) * (u32::MAX as u64);
        assert_eq!(lo.word(0), expected);
        assert_eq!(hi.word(0), 0);
    }

    #[test]
    fn zero_with_precision_of() {
        let modulus = fw(0xdeadbeef, 4);
        let z = FixedWidthBigUint::zero_with_precision_of(&modulus);
        assert!(z.is_zero());
        assert_eq!(z.n_limbs(), 4);
    }

    #[test]
    fn parity() {
        assert!(Parity::is_odd(fw(7, 1)));
        assert!(Parity::is_even(fw(8, 1)));
        assert!(Parity::is_odd(&fw(7, 1)));
    }

    #[test]
    fn from_u8_minimal_width() {
        let x = FixedWidthBigUint::from(5u8);
        assert_eq!(x.word(0), 5);
        assert_eq!(x.n_limbs(), 1);
    }

    #[test]
    fn ref_binops() {
        let a = fw(10, 2);
        let b = fw(3, 2);
        // &T - &T
        let r1 = &a - &b;
        assert_eq!(r1.word(0), 7);
        // T + &T
        let r2 = a.clone() + &b;
        assert_eq!(r2.word(0), 13);
        // T - &T
        let r3 = a.clone() - &b;
        assert_eq!(r3.word(0), 7);
        // T * &T
        let r4 = a.clone() * &b;
        assert_eq!(r4.word(0), 30);
    }
}
