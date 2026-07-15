// FixedWidthBigUint: const-num-traits + CiosRowOps for num-bigint.
//
// Design: stores a raw Vec<BigDigit> of exactly n_limbs digits (may have
// trailing zeros — the caller's invariant, NOT BigUint's invariant). Arithmetic
// converts to BigUint temporarily (which normalizes), then pads the result Vec
// back to n_limbs. This avoids all conflict with BigUint's normalization
// assertion (a.last() != Some(&0)).
//
// Personality: Nct only. BigUint is inherently variable-time; the subtle CT
// traits cannot be honestly satisfied.

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
        overflowing::{OverflowingAdd, OverflowingSub},
        wrapping::{WrappingAdd, WrappingMul, WrappingSub},
    },
    BorrowingSub, FromByteSlice, HasPersonality, Nct, One, Parity, ToBytes, Zero,
};
use modmath_cios::CiosRowOps;
use core::cmp::Ordering;
use core::ops::{
    Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, RemAssign, Shl, Shr, ShrAssign, Sub,
};

/// Bits per native digit.
const DIGIT_BITS: u32 = big_digit::BITS as u32;
/// Bytes per native digit.
const DIGIT_BYTES: usize = (DIGIT_BITS / 8) as usize;

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Build a normalized BigUint from a raw digit slice (removes trailing zeros).
fn biguint_from_digits(digits: &[BigDigit]) -> BigUint {
    let mut d = digits.to_vec();
    while d.last() == Some(&0) {
        d.pop();
    }
    BigUint { data: d }
}

/// Extract a BigUint's digits into a Vec padded to `n` entries with zeros.
fn digits_from_biguint(mut v: BigUint, n: usize) -> Vec<BigDigit> {
    v.data.resize(n, 0);
    v.data
}

/// The modular ceiling `2^(n * DIGIT_BITS)` as a BigUint.
fn modulus(n: usize) -> BigUint {
    BigUint::from(1u32) << (n * DIGIT_BITS as usize)
}

/// The bitmask for `n * DIGIT_BITS` bits.
fn low_mask(n: usize) -> BigUint {
    modulus(n) - BigUint::from(1u32)
}

// ── FixedWidthBigUint ─────────────────────────────────────────────────────────

/// A fixed-width unsigned integer backed by a stable [`Vec<BigDigit>`].
///
/// Unlike [`BigUint`] (which normalizes after every operation, removing leading
/// zero digits), `FixedWidthBigUint` keeps exactly `n_limbs` digits at all
/// times. This makes `word_count()` stable across arithmetic, which is required
/// by [`CiosRowOps`].
///
/// Arithmetic is performed by temporarily creating a normalized `BigUint` and
/// padding the result back to `n_limbs`.
///
/// Implements `Nct` personality for modmath's `constrained` and `strict` test
/// flavours. No `Ct` wrapper — `BigUint` is inherently variable-time.
pub struct FixedWidthBigUint {
    /// Raw digit slice, little-endian, always exactly `n_limbs` entries.
    /// May have trailing zeros (differs from BigUint's invariant).
    data: Vec<BigDigit>,
    n_limbs: usize,
}

impl FixedWidthBigUint {
    /// Construct from a `BigUint`, padding to `n_limbs`.
    pub fn new(value: BigUint, n_limbs: usize) -> Self {
        Self { data: digits_from_biguint(value, n_limbs), n_limbs }
    }

    /// Construct from a `u64` literal.
    pub fn from_u64(value: u64, n_limbs: usize) -> Self {
        Self::new(BigUint::from(value), n_limbs)
    }

    /// The declared limb count (stable across all operations).
    #[inline]
    pub fn n_limbs(&self) -> usize {
        self.n_limbs
    }

    /// Convert to a normalized `BigUint` for arithmetic.
    #[inline]
    fn to_biguint(&self) -> BigUint {
        biguint_from_digits(&self.data)
    }

    /// Consume a BigUint arithmetic result, padding back to `n_limbs`.
    #[inline]
    fn from_biguint_result(&self, v: BigUint) -> Self {
        Self { data: digits_from_biguint(v, self.n_limbs), n_limbs: self.n_limbs }
    }
}

// ── Clone / Debug / Hash ──────────────────────────────────────────────────────

impl Clone for FixedWidthBigUint {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), n_limbs: self.n_limbs }
    }
}

impl core::fmt::Debug for FixedWidthBigUint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FixedWidthBigUint({:?}, n_limbs={})", &self.data, self.n_limbs)
    }
}

impl core::hash::Hash for FixedWidthBigUint {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
        self.n_limbs.hash(state);
    }
}

// ── Equality / Ordering ───────────────────────────────────────────────────────
// Little-endian: most significant digit is last. Compare from the top down.

impl PartialEq for FixedWidthBigUint {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for FixedWidthBigUint {}

impl Ord for FixedWidthBigUint {
    fn cmp(&self, other: &Self) -> Ordering {
        let n = self.n_limbs.max(other.n_limbs);
        for i in (0..n).rev() {
            let a = self.data.get(i).copied().unwrap_or(0);
            let b = other.data.get(i).copied().unwrap_or(0);
            match a.cmp(&b) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}
impl PartialOrd for FixedWidthBigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Default for FixedWidthBigUint {
    fn default() -> Self {
        Self { data: Vec::new(), n_limbs: 0 }
    }
}

// ── From<uN> — minimal-width policy ──────────────────────────────────────────
// Constructs with the minimum number of limbs needed to hold the value.
// Modmath widens operands to the modulus width via zero_with_precision_of/
// widen_to_precision_of before arithmetic, so the starting width doesn't matter.

macro_rules! impl_from_uint {
    ($($t:ty),*) => {$(
        impl From<$t> for FixedWidthBigUint {
            fn from(v: $t) -> Self {
                let bu = BigUint::from(v);
                let n = (bu.data.len()).max(1);
                Self::new(bu, n)
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
        self.data.iter().all(|&d| d == 0)
    }
    fn set_zero(&mut self) {
        for d in self.data.iter_mut() {
            *d = 0;
        }
    }
}

impl One for FixedWidthBigUint {
    fn one() -> Self {
        // n_limbs=0; callers that need precision use one_with_precision.
        Self { data: vec![1], n_limbs: 0 }
    }
    fn is_one(&self) -> bool {
        self.data.first().copied() == Some(1)
            && self.data[1..].iter().all(|&d| d == 0)
    }
    fn set_one(&mut self) {
        if self.n_limbs > 0 {
            self.data[0] = 1;
            for d in self.data[1..].iter_mut() {
                *d = 0;
            }
        }
    }
}

// ── BitsPrecision / WithPrecision ─────────────────────────────────────────────

impl BitsPrecision for FixedWidthBigUint {
    fn bits_precision(&self) -> u32 {
        (self.n_limbs * DIGIT_BITS as usize) as u32
    }
}

impl WithPrecision for FixedWidthBigUint {
    /// Identity: fixed-width types can't change their width.
    fn widen_to_precision(self, _bits_precision: u32) -> Self {
        self
    }

    /// Override the default (which calls widen_to_precision = identity) to
    /// correctly construct a zero at the requested width.
    fn zero_with_precision(bits_precision: u32) -> Self
    where
        Self: Zero,
    {
        let n = bits_precision.div_ceil(DIGIT_BITS) as usize;
        Self { data: vec![0; n], n_limbs: n }
    }

    fn one_with_precision(bits_precision: u32) -> Self
    where
        Self: One,
    {
        let n = bits_precision.div_ceil(DIGIT_BITS) as usize;
        let mut data = vec![0; n];
        if n > 0 {
            data[0] = 1;
        }
        Self { data, n_limbs: n }
    }

    // Copy bound dropped in alpha.3 — now available for non-Copy heap carriers.
    fn widen_to_precision_of(self, witness: &Self) -> Self {
        // Fixed-width: can't grow a value; identity within the same n_limbs.
        // If witness is wider, re-express self at that width (zero-extend).
        let n = witness.n_limbs;
        Self { data: digits_from_biguint(self.to_biguint(), n), n_limbs: n }
    }

    fn zero_with_precision_of(witness: &Self) -> Self
    where
        Self: Zero,
    {
        Self { data: vec![0; witness.n_limbs], n_limbs: witness.n_limbs }
    }

    fn one_with_precision_of(witness: &Self) -> Self
    where
        Self: One,
    {
        let n = witness.n_limbs;
        let mut data = vec![0; n];
        if n > 0 {
            data[0] = 1;
        }
        Self { data, n_limbs: n }
    }
}

// ── Parity ────────────────────────────────────────────────────────────────────

impl Parity for FixedWidthBigUint {
    fn is_odd(self) -> bool {
        self.data.first().map_or(false, |d| d & 1 == 1)
    }
    fn is_even(self) -> bool {
        !Parity::is_odd(self)
    }
}

// ── Operator traits ───────────────────────────────────────────────────────────

impl Add for FixedWidthBigUint {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        let result = self.to_biguint() + rhs.to_biguint();
        Self { data: digits_from_biguint(result, n), n_limbs: n }
    }
}

impl Sub for FixedWidthBigUint {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        let result = self.to_biguint() - rhs.to_biguint();
        Self { data: digits_from_biguint(result, n), n_limbs: n }
    }
}

impl Mul for FixedWidthBigUint {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        let result = self.to_biguint() * rhs.to_biguint();
        Self { data: digits_from_biguint(result, n), n_limbs: n }
    }
}

impl Div for FixedWidthBigUint {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        let result = self.to_biguint() / rhs.to_biguint();
        Self { data: digits_from_biguint(result, n), n_limbs: n }
    }
}

impl Rem for FixedWidthBigUint {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        let result = self.to_biguint() % rhs.to_biguint();
        Self { data: digits_from_biguint(result, n), n_limbs: n }
    }
}

impl RemAssign for FixedWidthBigUint {
    fn rem_assign(&mut self, rhs: Self) {
        let result = self.to_biguint() % rhs.to_biguint();
        self.data = digits_from_biguint(result, self.n_limbs);
    }
}

impl RemAssign<&FixedWidthBigUint> for FixedWidthBigUint {
    fn rem_assign(&mut self, rhs: &Self) {
        let result = self.to_biguint() % rhs.to_biguint();
        self.data = digits_from_biguint(result, self.n_limbs);
    }
}

// &T op &T — for modmath constrained for<'a> bounds
impl Rem<&FixedWidthBigUint> for &FixedWidthBigUint {
    type Output = FixedWidthBigUint;
    fn rem(self, rhs: &FixedWidthBigUint) -> FixedWidthBigUint {
        let n = self.n_limbs;
        let result = self.to_biguint() % rhs.to_biguint();
        FixedWidthBigUint { data: digits_from_biguint(result, n), n_limbs: n }
    }
}

impl Div<&FixedWidthBigUint> for &FixedWidthBigUint {
    type Output = FixedWidthBigUint;
    fn div(self, rhs: &FixedWidthBigUint) -> FixedWidthBigUint {
        let n = self.n_limbs;
        let result = self.to_biguint() / rhs.to_biguint();
        FixedWidthBigUint { data: digits_from_biguint(result, n), n_limbs: n }
    }
}

impl Sub<FixedWidthBigUint> for &FixedWidthBigUint {
    type Output = FixedWidthBigUint;
    fn sub(self, rhs: FixedWidthBigUint) -> FixedWidthBigUint {
        let n = self.n_limbs;
        let result = self.to_biguint() - rhs.to_biguint();
        FixedWidthBigUint { data: digits_from_biguint(result, n), n_limbs: n }
    }
}

impl Shr<usize> for FixedWidthBigUint {
    type Output = Self;
    fn shr(self, n: usize) -> Self {
        let limbs = self.n_limbs;
        let result = self.to_biguint() >> n;
        Self { data: digits_from_biguint(result, limbs), n_limbs: limbs }
    }
}

impl ShrAssign<usize> for FixedWidthBigUint {
    fn shr_assign(&mut self, n: usize) {
        let result = self.to_biguint() >> n;
        self.data = digits_from_biguint(result, self.n_limbs);
    }
}

impl Shl<usize> for FixedWidthBigUint {
    type Output = Self;
    fn shl(self, n: usize) -> Self {
        let limbs = self.n_limbs;
        let result = self.to_biguint() << n;
        Self { data: digits_from_biguint(result, limbs), n_limbs: limbs }
    }
}

impl BitAnd for FixedWidthBigUint {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        // Direct digit-level AND — no BigUint conversion needed.
        let mut data = vec![0; n];
        for i in 0..n {
            data[i] = self.data.get(i).copied().unwrap_or(0)
                & rhs.data.get(i).copied().unwrap_or(0);
        }
        Self { data, n_limbs: n }
    }
}

impl BitOr for FixedWidthBigUint {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        let mut data = vec![0; n];
        for i in 0..n {
            data[i] = self.data.get(i).copied().unwrap_or(0)
                | rhs.data.get(i).copied().unwrap_or(0);
        }
        Self { data, n_limbs: n }
    }
}

impl BitXor for FixedWidthBigUint {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        let mut data = vec![0; n];
        for i in 0..n {
            data[i] = self.data.get(i).copied().unwrap_or(0)
                ^ rhs.data.get(i).copied().unwrap_or(0);
        }
        Self { data, n_limbs: n }
    }
}

impl BitAnd for &FixedWidthBigUint {
    type Output = FixedWidthBigUint;
    fn bitand(self, rhs: Self) -> FixedWidthBigUint {
        let n = self.n_limbs;
        let mut data = vec![0; n];
        for i in 0..n {
            data[i] = self.data.get(i).copied().unwrap_or(0)
                & rhs.data.get(i).copied().unwrap_or(0);
        }
        FixedWidthBigUint { data, n_limbs: n }
    }
}

// ── Wrapping arithmetic ───────────────────────────────────────────────────────

impl WrappingAdd for FixedWidthBigUint {
    type Output = Self;
    fn wrapping_add(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        let width = n * DIGIT_BITS as usize;
        let sum = self.to_biguint() + rhs.to_biguint();
        let result = if sum.bits() > width as u64 {
            sum & low_mask(n)
        } else {
            sum
        };
        Self { data: digits_from_biguint(result, n), n_limbs: n }
    }
}

impl WrappingSub for FixedWidthBigUint {
    type Output = Self;
    fn wrapping_sub(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        let lhs = self.to_biguint();
        let rhs_b = rhs.to_biguint();
        let result = if lhs >= rhs_b {
            lhs - rhs_b
        } else {
            modulus(n) + lhs - rhs_b
        };
        Self { data: digits_from_biguint(result, n), n_limbs: n }
    }
}

impl WrappingMul for FixedWidthBigUint {
    type Output = Self;
    fn wrapping_mul(self, rhs: Self) -> Self {
        let n = self.n_limbs;
        let product = self.to_biguint() * rhs.to_biguint();
        let result = product & low_mask(n);
        Self { data: digits_from_biguint(result, n), n_limbs: n }
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
        let n = self.n_limbs;
        let width = n * DIGIT_BITS as usize;
        let sum = self.to_biguint() + rhs.to_biguint();
        let overflow = sum.bits() > width as u64;
        let result = if overflow { sum & low_mask(n) } else { sum };
        (Self { data: digits_from_biguint(result, n), n_limbs: n }, overflow)
    }
}

impl OverflowingSub for FixedWidthBigUint {
    type Output = Self;
    fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let n = self.n_limbs;
        let lhs = self.to_biguint();
        let rhs_b = rhs.to_biguint();
        if lhs >= rhs_b {
            let r = lhs - rhs_b;
            (Self { data: digits_from_biguint(r, n), n_limbs: n }, false)
        } else {
            let r = modulus(n) + lhs - rhs_b;
            (Self { data: digits_from_biguint(r, n), n_limbs: n }, true)
        }
    }
}

// ── Checked arithmetic ────────────────────────────────────────────────────────

impl CheckedAdd for FixedWidthBigUint {
    type Output = Self;
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let n = self.n_limbs;
        let result = self.to_biguint() + rhs.to_biguint();
        Some(Self { data: digits_from_biguint(result, n), n_limbs: n })
    }
}

impl CheckedMul for FixedWidthBigUint {
    type Output = Self;
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        let n = self.n_limbs;
        let result = self.to_biguint() * rhs.to_biguint();
        Some(Self { data: digits_from_biguint(result, n), n_limbs: n })
    }
}

// ── BorrowingSub ──────────────────────────────────────────────────────────────

impl BorrowingSub for FixedWidthBigUint {
    type Output = Self;
    fn borrowing_sub(self, rhs: Self, borrow: bool) -> (Self, bool) {
        let n = self.n_limbs;
        let lhs = BigInt::from(self.to_biguint());
        let rhs_int = BigInt::from(rhs.to_biguint()) + BigInt::from(borrow as u32);
        let diff = lhs - rhs_int;
        let (result, borrow_out) = if diff < BigInt::from(0i32) {
            let wrapped = (diff + BigInt::from(modulus(n)))
                .to_biguint()
                .expect("borrow_sub: modular wrap must be non-negative");
            (wrapped, true)
        } else {
            (diff.to_biguint().expect("borrow_sub: non-negative result"), false)
        };
        (Self { data: digits_from_biguint(result, n), n_limbs: n }, borrow_out)
    }
}

// ── CarryingMul ───────────────────────────────────────────────────────────────

impl CarryingMul for FixedWidthBigUint {
    type Unsigned = Self;
    type Output = Self;

    fn carrying_mul(self, rhs: Self, carry: Self) -> (Self, Self) {
        let n = self.n_limbs;
        let width = n * DIGIT_BITS as usize;
        let product = self.to_biguint() * rhs.to_biguint() + carry.to_biguint();
        let mask = low_mask(n);
        let lo = product.clone() & &mask;
        let hi = product >> width;
        (
            Self { data: digits_from_biguint(lo, n), n_limbs: n },
            Self { data: digits_from_biguint(hi, n), n_limbs: n },
        )
    }

    fn carrying_mul_add(self, rhs: Self, carry: Self, add: Self) -> (Self, Self) {
        let n = self.n_limbs;
        let width = n * DIGIT_BITS as usize;
        let product =
            self.to_biguint() * rhs.to_biguint() + carry.to_biguint() + add.to_biguint();
        let mask = low_mask(n);
        let lo = product.clone() & &mask;
        let hi = product >> width;
        (
            Self { data: digits_from_biguint(lo, n), n_limbs: n },
            Self { data: digits_from_biguint(hi, n), n_limbs: n },
        )
    }
}

// ── ToBytes / FromByteSlice ───────────────────────────────────────────────────

impl ToBytes for FixedWidthBigUint {
    type Bytes = Vec<u8>;
    fn to_be_bytes(self) -> Vec<u8> {
        let expected = self.n_limbs * DIGIT_BYTES;
        let bytes = self.to_biguint().to_bytes_be();
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
        let mut bytes = self.to_biguint().to_bytes_le();
        bytes.resize(expected, 0);
        bytes
    }
}

impl FromByteSlice for FixedWidthBigUint {
    fn from_be_slice(bytes: &[u8]) -> Result<Self, ByteSliceError> {
        let n = bytes.len().div_ceil(DIGIT_BYTES);
        Ok(Self::new(BigUint::from_bytes_be(bytes), n))
    }
    fn from_le_slice(bytes: &[u8]) -> Result<Self, ByteSliceError> {
        let n = bytes.len().div_ceil(DIGIT_BYTES);
        Ok(Self::new(BigUint::from_bytes_le(bytes), n))
    }
}

// ── CiosRowOps ────────────────────────────────────────────────────────────────
//
// Operates directly on self.data — the raw digit Vec — with no BigUint
// conversion. word_count() returns the declared n_limbs (never changes).
// word(i) returns data[i] or 0 if i >= data.len() (shouldn't happen since
// data.len() == n_limbs, but belt-and-suspenders).
//
// mul_acc_row / mul_acc_shift_row: schoolbook inner row using u128 for the
// double-wide product. Semantics match the bnum CiosRowOps exactly.

impl CiosRowOps for FixedWidthBigUint {
    type Word = crate::Digit;

    #[inline]
    fn word_count(&self) -> usize {
        self.n_limbs
    }

    #[inline]
    fn word(&self, i: usize) -> crate::Digit {
        self.data.get(i).copied().unwrap_or(0)
    }

    fn mul_acc_row(
        scalar: crate::Digit,
        multiplicand: &Self,
        acc: &mut Self,
        carry_in: crate::Digit,
    ) -> crate::Digit {
        let n = multiplicand.n_limbs;
        let mut carry = carry_in as u128;
        let s = scalar as u128;
        let mut j = 0;
        while j < n {
            let m = multiplicand.data.get(j).copied().unwrap_or(0) as u128;
            let a = acc.data.get(j).copied().unwrap_or(0) as u128;
            let product = s * m + a + carry;
            acc.data[j] = product as crate::Digit;
            carry = product >> DIGIT_BITS;
            j += 1;
        }
        carry as crate::Digit
    }

    fn mul_acc_shift_row(
        scalar: crate::Digit,
        multiplicand: &Self,
        acc: &mut Self,
        acc_hi: crate::Digit,
    ) -> crate::Digit {
        let n = multiplicand.n_limbs;
        let s = scalar as u128;

        // Word 0: discard the low word (the shifted-out digit).
        let p0 = s * multiplicand.data.get(0).copied().unwrap_or(0) as u128
            + acc.data[0] as u128;
        let mut carry = (p0 >> DIGIT_BITS) as u64;

        // Words 1..n: accumulate and shift down by one position.
        let mut j = 1;
        while j < n {
            let m = multiplicand.data.get(j).copied().unwrap_or(0) as u128;
            let a = acc.data[j] as u128;
            let product = s * m + a + carry as u128;
            acc.data[j - 1] = product as crate::Digit;
            carry = (product >> DIGIT_BITS) as u64;
            j += 1;
        }

        // Top slot: acc_hi + carry; return overflow bit.
        let (sum, overflow) = acc_hi.overflowing_add(carry);
        if n > 0 {
            acc.data[n - 1] = sum;
        }
        overflow as crate::Digit
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use const_num_traits::ops::bits::BitsPrecision;
    use const_num_traits::{One, Zero, WithPrecision};

    fn fw(value: u64, n: usize) -> FixedWidthBigUint {
        FixedWidthBigUint::from_u64(value, n)
    }

    #[test]
    fn word_count_stable_after_sub() {
        // Subtract to produce a small result; word_count must still return n_limbs.
        let a = fw(100, 2);
        let b = fw(99, 2);
        let c = a - b;
        assert_eq!(c.n_limbs(), 2);
        assert_eq!(c.word_count(), 2);
        assert_eq!(c.word(0), 1);
        assert_eq!(c.word(1), 0); // zero-padded
    }

    #[test]
    fn bits_precision_stable() {
        let x = fw(0, 4);
        assert_eq!(x.bits_precision(), 4 * DIGIT_BITS);
    }

    #[test]
    fn wrapping_add_truncates() {
        // Digit::MAX + 1 should wrap to 0 for a 1-limb value.
        let max = fw(crate::Digit::MAX, 1);
        let one = fw(1, 1);
        let result = max.wrapping_add(one);
        assert_eq!(result.word(0), 0);
        assert_eq!(result.n_limbs(), 1);
    }

    #[test]
    fn wrapping_sub_modular() {
        // 0 - 1 should wrap to Digit::MAX for a 1-limb value.
        let zero = fw(0, 1);
        let one = fw(1, 1);
        let result = zero.wrapping_sub(one);
        assert_eq!(result.word(0), crate::Digit::MAX);
    }

    #[test]
    fn overflowing_sub_flag() {
        let (_, overflow) = fw(5, 1).overflowing_sub(fw(10, 1));
        assert!(overflow);
        let (diff, no_overflow) = fw(10, 1).overflowing_sub(fw(5, 1));
        assert!(!no_overflow);
        assert_eq!(diff.word(0), 5);
    }

    #[test]
    fn carrying_mul_splits() {
        // (2^32-1)^2 = 2^64 - 2^33 + 1 — fits in 64 bits, hi = 0.
        let a = fw(u32::MAX as u64, 1);
        let b = fw(u32::MAX as u64, 1);
        let (lo, hi) = a.carrying_mul(b, fw(0, 1));
        let expected: u64 = (u32::MAX as u64) * (u32::MAX as u64);
        assert_eq!(lo.word(0), expected);
        assert_eq!(hi.word(0), 0);
    }

    #[test]
    fn zero_with_precision_is_zero() {
        let z = FixedWidthBigUint::zero_with_precision(128);
        assert_eq!(z.n_limbs(), 128u32.div_ceil(DIGIT_BITS) as usize);
        assert!(z.is_zero());
    }

    #[test]
    fn one_with_precision() {
        let o = FixedWidthBigUint::one_with_precision(64);
        assert_eq!(o.word(0), 1);
        assert!(!o.is_zero());
    }

    #[test]
    fn parity() {
        assert!(Parity::is_odd(fw(7, 1)));
        assert!(Parity::is_even(fw(8, 1)));
    }

    #[test]
    fn borrowing_sub() {
        let (result, borrow) = fw(10, 1).borrowing_sub(fw(3, 1), false);
        assert_eq!(result.word(0), 7);
        assert!(!borrow);

        let (_, borrow) = fw(3, 1).borrowing_sub(fw(10, 1), false);
        assert!(borrow);
    }

    #[test]
    fn is_zero_after_zero_with_precision() {
        let z = FixedWidthBigUint::zero_with_precision(256);
        assert!(z.is_zero());
        assert_eq!(z.word_count(), 256u32.div_ceil(DIGIT_BITS) as usize);
    }
}
