// Copyright (c) 2019-2026 Provable Inc.
// This file is part of the snarkVM library.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! An eight-lane AVX-512 IFMA backend for BLS12-377 `Fq` arithmetic.
//!
//! The scalar `Fp384` representation is Montgomery form with `R = 2^384` over
//! six 64-bit limbs. `vpmadd52{lo,hi}uq` multiplies 52-bit operands, so this
//! backend uses a second Montgomery domain with `R = 2^416` over eight 52-bit
//! limbs, and converts at the boundary. Elements are held structure-of-arrays:
//! `Fq8[j]` is limb `j` of all eight elements.
//!
//! Every operation here is required to agree bit-for-bit with the scalar path,
//! since the backend is selected at runtime and nodes with different CPUs must
//! agree. `tests` below enforce that against `snarkvm_curves::bls12_377::Fq`.
//!
//! This module is an approved exception to the crate's `warn(unsafe_code)`
//! policy: `core::arch` intrinsics are `unsafe` by definition. Every `unsafe`
//! entry point here is guarded by `#[target_feature]` and documents that the
//! caller must first check [`is_available`]. The same exception is taken by
//! `prefetch.rs` for `_mm_prefetch`.
#![allow(unsafe_code)]

use snarkvm_curves::bls12_377::{Fq, FqParameters};
use snarkvm_fields::{FieldParameters, Fp384};
use snarkvm_utilities::BigInteger384;

/// The number of field elements processed per vector operation.
pub const LANES: usize = 8;

/// The number of 52-bit limbs needed to hold a 377-bit modulus.
pub const LIMBS: usize = 8;

const MASK: u64 = (1 << 52) - 1;

/// The BLS12-377 base field modulus, in radix 2^52.
const P: [u64; LIMBS] = [
    0x8c00000000001,
    0x4430000000850,
    0xa094800170b5d,
    0x138f1ef3622fb,
    0xb1a22d9f300f5,
    0x3b05c06ca1493,
    0xa4617c510eac6,
    0x0000000001ae3,
];

/// `-P^{-1} mod 2^52`.
const N0INV: u64 = 0x8bfffffffffff;

/// `2^448 mod P`. Shifts a value from the scalar domain (`R = 2^384`) into this
/// one directly, without first reducing to a canonical integer.
const TO_IFMA: [u64; LIMBS] = [
    0x07ccefe7c5a25,
    0x9dee49ccfcf9a,
    0x3dc3ff79f2a81,
    0xb7eaa16af28b0,
    0xe5d18e3b07d04,
    0x85efd62fb7463,
    0x8c84314d2bbc1,
    0x0000000000197,
];

/// `2^384 mod P`, the inverse shift.
const FROM_IFMA: [u64; LIMBS] = [
    0xdffffffffff68,
    0x837fffffb102c,
    0xa7d3ff251409f,
    0x63059f7db3a98,
    0x87b4e97b76e7c,
    0xf495bf803c84e,
    0x661e2fdf49a4c,
    0x00000000008d6,
];

/// Returns `true` if this CPU supports the instructions this backend needs.
///
/// Note that AVX-512F is *not* sufficient: IFMA first appeared on Intel Ice
/// Lake and AMD Zen 4, so Skylake-SP and Cascade Lake Xeons report AVX-512F
/// while lacking `vpmadd52`.
#[inline]
pub fn is_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512ifma")
    })
}

/// Whether the MSM should dispatch to the vectorized path.
///
/// On wherever the CPU supports it. Setting `SNARKVM_DISABLE_AVX512_IFMA`
/// forces the scalar path, as an operator kill switch and for A/B measurement.
///
/// Correctness does not depend on this switch. Both paths produce identical
/// field elements, so two nodes that answer differently here still agree on
/// every result; the tests check that against the scalar path directly.
#[inline]
pub fn is_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| is_available() && std::env::var_os("SNARKVM_DISABLE_AVX512_IFMA").is_none())
}

// ---------------------------------------------------------------------------
// Limb regrouping between the scalar 6x64 representation and 8x52.
// ---------------------------------------------------------------------------

/// Regroups a 384-bit little-endian value from six 64-bit limbs into eight
/// 52-bit limbs.
#[inline]
const fn regroup_64_to_52(x: &[u64; 6]) -> [u64; LIMBS] {
    let mut out = [0u64; LIMBS];
    let mut i = 0;
    while i < LIMBS {
        let start = i * 52;
        let word = start / 64;
        let shift = start % 64;
        // `start + 52 <= 416`, and reading at most two words never runs past
        // index 5 because the top limb starts at bit 364 (word 5, shift 44).
        let mut v = x[word] >> shift;
        if shift > 12 && word + 1 < 6 {
            v |= x[word + 1] << (64 - shift);
        }
        out[i] = v & MASK;
        i += 1;
    }
    out
}

/// The inverse of [`regroup_64_to_52`].
#[inline]
fn regroup_52_to_64(x: &[u64; LIMBS]) -> [u64; 6] {
    let mut out = [0u64; 6];
    for (i, limb) in x.iter().enumerate() {
        let start = i * 52;
        let word = start / 64;
        let shift = start % 64;
        out[word] |= limb << shift;
        if shift > 12 && word + 1 < 6 {
            out[word + 1] |= limb >> (64 - shift);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Scalar reference implementation.
//
// This mirrors the vector kernel exactly and exists so the tests can localize a
// failure to either the algorithm or the intrinsics.
// ---------------------------------------------------------------------------

/// CIOS Montgomery multiplication in radix 2^52, returning a canonical result.
pub const fn mont_mul_ref(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
    let mut t = [0u64; LIMBS + 1];
    let mut i = 0;
    while i < LIMBS {
        let mut carry = 0u64;
        let mut j = 0;
        while j < LIMBS {
            let prod = (a[j] as u128) * (b[i] as u128) + (t[j] as u128) + (carry as u128);
            t[j] = (prod as u64) & MASK;
            carry = (prod >> 52) as u64;
            j += 1;
        }
        t[LIMBS] += carry;

        let m = t[0].wrapping_mul(N0INV) & MASK;
        let mut carry = 0u64;
        let mut j = 0;
        while j < LIMBS {
            let prod = (m as u128) * (P[j] as u128) + (t[j] as u128) + (carry as u128);
            t[j] = (prod as u64) & MASK;
            carry = (prod >> 52) as u64;
            j += 1;
        }
        debug_assert!(t[0] == 0);
        let mut j = 0;
        while j < LIMBS {
            t[j] = t[j + 1];
            j += 1;
        }
        t[LIMBS] = 0;
        t[LIMBS - 1] = t[LIMBS - 1].wrapping_add(carry);
        i += 1;
    }
    let mut out = [0u64; LIMBS];
    let mut j = 0;
    while j < LIMBS {
        out[j] = t[j];
        j += 1;
    }
    sub_p_if_ge(out)
}

/// Subtracts `P` from `x` if `x >= P`, leaving a canonical representative.
#[inline]
const fn sub_p_if_ge(x: [u64; LIMBS]) -> [u64; LIMBS] {
    let mut d = [0u64; LIMBS];
    let mut borrow = 0i128;
    let mut j = 0;
    while j < LIMBS {
        let v = (x[j] as i128) - (P[j] as i128) - borrow;
        d[j] = (v as u64) & MASK;
        borrow = (v < 0) as i128;
        j += 1;
    }
    if borrow == 0 { d } else { x }
}

/// Adds `P` to `x`. Used to correct an underflowed subtraction.
#[inline]
fn add_p(x: &mut [u64; LIMBS]) {
    let mut carry = 0u64;
    for j in 0..LIMBS {
        let v = x[j] + P[j] + carry;
        x[j] = v & MASK;
        carry = v >> 52;
    }
}

/// Modular addition of canonical inputs.
pub fn add_ref(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
    let mut out = [0u64; LIMBS];
    let mut carry = 0u64;
    for j in 0..LIMBS {
        let v = a[j] + b[j] + carry;
        out[j] = v & MASK;
        carry = v >> 52;
    }
    sub_p_if_ge(out)
}

/// Modular subtraction of canonical inputs.
pub fn sub_ref(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
    let mut out = [0u64; LIMBS];
    let mut borrow = 0i128;
    for j in 0..LIMBS {
        let v = (a[j] as i128) - (b[j] as i128) - borrow;
        out[j] = (v as u64) & MASK;
        borrow = i128::from(v < 0);
    }
    if borrow != 0 {
        add_p(&mut out);
    }
    out
}

/// Re-chunks the raw Montgomery limbs of `x` into radix 2^52 without changing
/// the value. The result still carries the scalar domain's `R = 2^384`; a
/// multiply by [`TO_IFMA`] completes the move.
#[inline]
pub fn regroup_raw(x: &Fq) -> [u64; LIMBS] {
    regroup_64_to_52(&x.0.0)
}

/// Finishes the conversion started by [`regroup_raw`], one element at a time.
#[inline]
pub const fn shift_into_domain(raw: &[u64; LIMBS]) -> [u64; LIMBS] {
    mont_mul_ref(raw, &TO_IFMA)
}

/// The field element one, in this domain. Derived at compile time from the
/// scalar field's own `R`, so it cannot drift from the curve parameters.
pub const ONE: [u64; LIMBS] = shift_into_domain(&regroup_64_to_52(&FqParameters::R.0));

/// The field element one half, in this domain.
///
/// Montgomery representation is linear in the value it stands for, so halving
/// the representative halves the field element.
pub const HALF: [u64; LIMBS] = half_of(&ONE);

/// Halves a canonical value modulo `P`. Adding `P` to an odd value makes it
/// even without changing what it represents, so the shift stays exact.
const fn half_of(v: &[u64; LIMBS]) -> [u64; LIMBS] {
    let mut t = *v;
    if t[0] & 1 == 1 {
        let mut carry = 0u64;
        let mut j = 0;
        while j < LIMBS {
            let sum = t[j] + P[j] + carry;
            t[j] = sum & MASK;
            carry = sum >> 52;
            j += 1;
        }
    }
    let mut out = [0u64; LIMBS];
    let mut j = 0;
    while j < LIMBS {
        let carry_in = if j + 1 < LIMBS { t[j + 1] & 1 } else { 0 };
        out[j] = (t[j] >> 1) | (carry_in << 51);
        j += 1;
    }
    out
}

/// Converts a scalar `Fq` into this backend's Montgomery domain.
pub fn to_ifma(x: &Fq) -> [u64; LIMBS] {
    shift_into_domain(&regroup_raw(x))
}

/// Converts back into a scalar `Fq`.
pub fn from_ifma(x: &[u64; LIMBS]) -> Fq {
    // Shifting by `FROM_IFMA` lands directly on the scalar Montgomery form, so
    // the result can be rebuilt without a second conversion.
    Fp384::<FqParameters>(BigInteger384(regroup_52_to_64(&mont_mul_ref(x, &FROM_IFMA))), core::marker::PhantomData)
}

/// Converts eight elements out of the domain at once.
///
/// The per-element scalar path costs a full 8-limb Montgomery multiply, which
/// is far more expensive than the native 6-limb one; doing the shift eight at a
/// time keeps it off the critical path.
pub fn from_ifma_x8(x: &[[u64; LIMBS]; LANES]) -> [Fq; LANES] {
    if is_available() {
        // SAFETY: guarded by `is_available`.
        let raw = unsafe { mul_x8(&Fq8::load(x), &Fq8::load(&[FROM_IFMA; LANES])).store() };
        std::array::from_fn(|l| {
            Fp384::<FqParameters>(BigInteger384(regroup_52_to_64(&raw[l])), core::marker::PhantomData)
        })
    } else {
        std::array::from_fn(|l| from_ifma(&x[l]))
    }
}

/// Converts eight elements into the domain at once.
pub fn to_ifma_x8(x: &[Fq; LANES]) -> [[u64; LIMBS]; LANES] {
    let raw: [[u64; LIMBS]; LANES] = std::array::from_fn(|l| regroup_raw(&x[l]));
    if is_available() {
        // SAFETY: guarded by `is_available`.
        unsafe { mul_x8(&Fq8::load(&raw), &Fq8::load(&[TO_IFMA; LANES])).store() }
    } else {
        std::array::from_fn(|l| shift_into_domain(&raw[l]))
    }
}

/// Moves a whole slice into the domain in place, eight elements at a time.
/// Each element must already have passed through [`regroup_raw`].
pub fn shift_into_domain_slice(raw: &mut [[u64; LIMBS]]) {
    if is_available() {
        let mut chunks = raw.chunks_exact_mut(LANES);
        for chunk in &mut chunks {
            let staged: [[u64; LIMBS]; LANES] = std::array::from_fn(|l| chunk[l]);
            // SAFETY: `is_available` was checked above.
            let converted = unsafe { mul_x8(&Fq8::load(&staged), &Fq8::load(&[TO_IFMA; LANES])).store() };
            chunk.copy_from_slice(&converted);
        }
        for elem in chunks.into_remainder() {
            *elem = shift_into_domain(elem);
        }
    } else {
        for elem in raw.iter_mut() {
            *elem = shift_into_domain(elem);
        }
    }
}

// ---------------------------------------------------------------------------
// Eight-lane vector kernel.
// ---------------------------------------------------------------------------

/// Eight field elements, structure-of-arrays: `.0[j]` holds limb `j` of each.
#[derive(Copy, Clone)]
#[repr(C, align(64))]
pub struct Fq8(pub [std::arch::x86_64::__m512i; LIMBS]);

impl Fq8 {
    /// Loads eight field elements into vector registers.
    ///
    /// # Safety
    /// Requires AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn load(elems: &[[u64; LIMBS]; LANES]) -> Self {
        use std::arch::x86_64::*;
        Self(std::array::from_fn(|j| {
            _mm512_set_epi64(
                elems[7][j] as i64,
                elems[6][j] as i64,
                elems[5][j] as i64,
                elems[4][j] as i64,
                elems[3][j] as i64,
                elems[2][j] as i64,
                elems[1][j] as i64,
                elems[0][j] as i64,
            )
        }))
    }

    /// Stores eight field elements back out.
    ///
    /// # Safety
    /// Requires AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn store(&self) -> [[u64; LIMBS]; LANES] {
        unsafe {
            use std::arch::x86_64::*;
            let mut out = [[0u64; LIMBS]; LANES];
            for (j, limb) in self.0.iter().enumerate() {
                let mut buf = [0i64; LANES];
                _mm512_storeu_si512(buf.as_mut_ptr() as *mut _, *limb);
                for (lane, v) in buf.iter().enumerate() {
                    out[lane][j] = *v as u64;
                }
            }
            out
        }
    }
}

/// Eight independent Montgomery multiplications.
///
/// # Safety
/// Requires AVX-512F and AVX512-IFMA; check [`is_available`] first.
#[target_feature(enable = "avx512f,avx512ifma")]
pub unsafe fn mul_x8(a: &Fq8, b: &Fq8) -> Fq8 {
    unsafe {
        use std::arch::x86_64::*;

        let mask = _mm512_set1_epi64(MASK as i64);
        let n0 = _mm512_set1_epi64(N0INV as i64);
        let zero = _mm512_setzero_si512();
        let p: [__m512i; LIMBS] = std::array::from_fn(|i| _mm512_set1_epi64(P[i] as i64));

        let mut t = [zero; LIMBS + 1];
        for i in 0..LIMBS {
            let bi = b.0[i];

            // t += a * b[i]. Each 52x52 product is split across a low and a high
            // instruction; the high halves belong to the next limb up.
            let mut lo = [zero; LIMBS];
            let mut hi = [zero; LIMBS];
            for j in 0..LIMBS {
                lo[j] = _mm512_madd52lo_epu64(t[j], a.0[j], bi);
                hi[j] = _mm512_madd52hi_epu64(zero, a.0[j], bi);
            }
            t[0] = lo[0];
            for j in 1..LIMBS {
                t[j] = _mm512_add_epi64(lo[j], hi[j - 1]);
            }
            t[LIMBS] = _mm512_add_epi64(t[LIMBS], hi[LIMBS - 1]);

            // m := t[0] * N0INV mod 2^52, the multiple of P that clears limb zero.
            let m = _mm512_and_si512(_mm512_madd52lo_epu64(zero, t[0], n0), mask);

            let mut lo2 = [zero; LIMBS];
            let mut hi2 = [zero; LIMBS];
            for j in 0..LIMBS {
                lo2[j] = _mm512_madd52lo_epu64(t[j], m, p[j]);
                hi2[j] = _mm512_madd52hi_epu64(zero, m, p[j]);
            }

            // Limb zero is now a multiple of 2^52, so shift the accumulator down.
            let carry0 = _mm512_srli_epi64(lo2[0], 52);
            let mut nt = [zero; LIMBS];
            for j in 0..LIMBS - 1 {
                let mut v = _mm512_add_epi64(lo2[j + 1], hi2[j]);
                if j == 0 {
                    v = _mm512_add_epi64(v, carry0);
                }
                nt[j] = v;
            }
            nt[LIMBS - 1] = _mm512_add_epi64(t[LIMBS], hi2[LIMBS - 1]);

            // Renormalize so every limb is back inside 52 bits.
            let mut carry = zero;
            for j in 0..LIMBS {
                let v = _mm512_add_epi64(nt[j], carry);
                carry = _mm512_srli_epi64(v, 52);
                t[j] = _mm512_and_si512(v, mask);
            }
            t[LIMBS] = carry;
        }

        let mut out = Fq8(std::array::from_fn(|j| t[j]));
        reduce_x8(&mut out);
        out
    }
}

/// Conditionally subtracts `P` in each lane, producing canonical limbs.
///
/// # Safety
/// Requires AVX-512F.
#[target_feature(enable = "avx512f")]
unsafe fn reduce_x8(x: &mut Fq8) {
    use std::arch::x86_64::*;
    let mask = _mm512_set1_epi64(MASK as i64);
    let p: [__m512i; LIMBS] = std::array::from_fn(|i| _mm512_set1_epi64(P[i] as i64));

    // Subtract P with borrow propagation. Limbs hold 52 bits inside 64-bit
    // lanes, so a negative intermediate sets bit 63 and shifting it down by 63
    // yields the borrow.
    let mut d = [_mm512_setzero_si512(); LIMBS];
    let mut borrow = _mm512_setzero_si512();
    for j in 0..LIMBS {
        let v = _mm512_sub_epi64(_mm512_sub_epi64(x.0[j], p[j]), borrow);
        borrow = _mm512_and_si512(_mm512_srli_epi64(v, 63), _mm512_set1_epi64(1));
        d[j] = _mm512_and_si512(v, mask);
    }
    // Keep the difference only where no final borrow occurred, i.e. x >= P.
    let keep = _mm512_cmpeq_epi64_mask(borrow, _mm512_setzero_si512());
    for (limb, diff) in x.0.iter_mut().zip(d.iter()) {
        *limb = _mm512_mask_blend_epi64(keep, *limb, *diff);
    }
}

/// Eight independent modular additions of canonical inputs.
///
/// # Safety
/// Requires AVX-512F.
#[target_feature(enable = "avx512f")]
pub unsafe fn add_x8(a: &Fq8, b: &Fq8) -> Fq8 {
    unsafe {
        use std::arch::x86_64::*;
        let mask = _mm512_set1_epi64(MASK as i64);
        let mut out = Fq8([_mm512_setzero_si512(); LIMBS]);
        let mut carry = _mm512_setzero_si512();
        for j in 0..LIMBS {
            let v = _mm512_add_epi64(_mm512_add_epi64(a.0[j], b.0[j]), carry);
            carry = _mm512_srli_epi64(v, 52);
            out.0[j] = _mm512_and_si512(v, mask);
        }
        reduce_x8(&mut out);
        out
    }
}

/// Eight independent modular subtractions of canonical inputs.
///
/// # Safety
/// Requires AVX-512F.
#[target_feature(enable = "avx512f")]
pub unsafe fn sub_x8(a: &Fq8, b: &Fq8) -> Fq8 {
    use std::arch::x86_64::*;
    let mask = _mm512_set1_epi64(MASK as i64);
    let one = _mm512_set1_epi64(1);
    let p: [__m512i; LIMBS] = std::array::from_fn(|i| _mm512_set1_epi64(P[i] as i64));

    let mut out = Fq8([_mm512_setzero_si512(); LIMBS]);
    let mut borrow = _mm512_setzero_si512();
    for j in 0..LIMBS {
        let v = _mm512_sub_epi64(_mm512_sub_epi64(a.0[j], b.0[j]), borrow);
        borrow = _mm512_and_si512(_mm512_srli_epi64(v, 63), one);
        out.0[j] = _mm512_and_si512(v, mask);
    }
    // Where the subtraction underflowed, add P back.
    let underflowed = _mm512_cmpeq_epi64_mask(borrow, one);
    let mut carry = _mm512_setzero_si512();
    for (limb, pj) in out.0.iter_mut().zip(p.iter()) {
        let addend = _mm512_maskz_mov_epi64(underflowed, *pj);
        let v = _mm512_add_epi64(_mm512_add_epi64(*limb, addend), carry);
        carry = _mm512_srli_epi64(v, 52);
        *limb = _mm512_and_si512(v, mask);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkvm_fields::{One, PrimeField, Zero};
    use snarkvm_utilities::{TestRng, Uniform};

    const ITERATIONS: usize = 5000;

    fn skip_if_unsupported() -> bool {
        if !is_available() {
            eprintln!("skipping: CPU lacks AVX512-IFMA");
            return true;
        }
        false
    }

    #[test]
    fn test_limb_regrouping_roundtrips() {
        let mut rng = TestRng::default();
        for _ in 0..ITERATIONS {
            let f = Fq::rand(&mut rng);
            let repr = f.to_bigint().0;
            assert_eq!(regroup_52_to_64(&regroup_64_to_52(&repr)), repr);
        }
    }

    #[test]
    fn test_domain_conversion_roundtrips() {
        let mut rng = TestRng::default();
        for _ in 0..ITERATIONS {
            let f = Fq::rand(&mut rng);
            assert_eq!(from_ifma(&to_ifma(&f)), f);
        }
        assert_eq!(from_ifma(&to_ifma(&Fq::zero())), Fq::zero());
        assert_eq!(from_ifma(&to_ifma(&Fq::one())), Fq::one());
    }

    #[test]
    fn test_domain_constants() {
        // Derived at compile time; check them against the runtime conversion
        // and against their closed forms, 2^416 and 2^415 mod P.
        assert_eq!(ONE, to_ifma(&Fq::one()));
        assert_eq!(HALF, to_ifma(&<Fq as snarkvm_fields::Field>::half()));
        assert_eq!(ONE, [
            0x33f67abdcbccf,
            0x13a714cde2ff1,
            0x80ef04efd57bd,
            0x68acd5a37c3a6,
            0x8c63c8655a9bd,
            0x7b4891129d194,
            0xb78d3aa6ef92f,
            0x0000000000028
        ]);
        assert_eq!(HALF, [
            0xdffb3d5ee5e68,
            0x2beb8a66f1c20,
            0x10c1c278a318d,
            0x3e1dfa4b6f351,
            0x1f02fb0245559,
            0xdb2728bf9f314,
            0x2df75b7bff1fa,
            0x0000000000d86
        ]);
        assert_eq!(add_ref(&HALF, &HALF), ONE);
        // `ONE` really is the multiplicative identity in this domain.
        let x = to_ifma(&Fq::from(123456789u64));
        assert_eq!(mont_mul_ref(&x, &ONE), x);
    }

    #[test]
    fn test_scalar_reference_matches_fq() {
        let mut rng = TestRng::default();
        for _ in 0..ITERATIONS {
            let a = Fq::rand(&mut rng);
            let b = Fq::rand(&mut rng);
            assert_eq!(from_ifma(&mont_mul_ref(&to_ifma(&a), &to_ifma(&b))), a * b);
        }
    }

    #[test]
    fn test_mul_x8_matches_fq() {
        if skip_if_unsupported() {
            return;
        }
        let mut rng = TestRng::default();
        for _ in 0..ITERATIONS / LANES {
            let a: [Fq; LANES] = std::array::from_fn(|_| Fq::rand(&mut rng));
            let b: [Fq; LANES] = std::array::from_fn(|_| Fq::rand(&mut rng));
            let av: [[u64; LIMBS]; LANES] = std::array::from_fn(|i| to_ifma(&a[i]));
            let bv: [[u64; LIMBS]; LANES] = std::array::from_fn(|i| to_ifma(&b[i]));
            let got = unsafe {
                let r = mul_x8(&Fq8::load(&av), &Fq8::load(&bv));
                r.store()
            };
            for (lane, r) in got.iter().enumerate() {
                assert_eq!(from_ifma(r), a[lane] * b[lane], "lane {lane}");
            }
        }
    }

    #[test]
    fn test_mul_x8_edge_cases() {
        if skip_if_unsupported() {
            return;
        }
        // Zero, one, and P-1 exercise the conditional subtraction.
        let p_minus_one = -Fq::one();
        let cases = [Fq::zero(), Fq::one(), p_minus_one, p_minus_one * p_minus_one];
        for a in cases {
            for b in cases {
                let av: [[u64; LIMBS]; LANES] = [to_ifma(&a); LANES];
                let bv: [[u64; LIMBS]; LANES] = [to_ifma(&b); LANES];
                let got = unsafe {
                    let r = mul_x8(&Fq8::load(&av), &Fq8::load(&bv));
                    r.store()
                };
                for (lane, r) in got.iter().enumerate() {
                    assert_eq!(from_ifma(r), a * b, "{a} * {b} lane {lane}");
                }
            }
        }
    }

    #[test]
    fn test_add_sub_x8_match_fq() {
        if skip_if_unsupported() {
            return;
        }
        let mut rng = TestRng::default();
        for _ in 0..ITERATIONS / LANES {
            let a: [Fq; LANES] = std::array::from_fn(|_| Fq::rand(&mut rng));
            let b: [Fq; LANES] = std::array::from_fn(|_| Fq::rand(&mut rng));
            let av: [[u64; LIMBS]; LANES] = std::array::from_fn(|i| to_ifma(&a[i]));
            let bv: [[u64; LIMBS]; LANES] = std::array::from_fn(|i| to_ifma(&b[i]));
            let (sum, diff) = unsafe {
                let (x, y) = (Fq8::load(&av), Fq8::load(&bv));
                (add_x8(&x, &y).store(), sub_x8(&x, &y).store())
            };
            for lane in 0..LANES {
                assert_eq!(from_ifma(&sum[lane]), a[lane] + b[lane], "add lane {lane}");
                assert_eq!(from_ifma(&diff[lane]), a[lane] - b[lane], "sub lane {lane}");
                // The scalar reference must agree too.
                assert_eq!(add_ref(&av[lane], &bv[lane]), sum[lane]);
                assert_eq!(sub_ref(&av[lane], &bv[lane]), diff[lane]);
            }
        }
    }

    #[test]
    fn test_add_sub_x8_edge_cases() {
        if skip_if_unsupported() {
            return;
        }
        let cases = [Fq::zero(), Fq::one(), -Fq::one(), -Fq::one() - Fq::one()];
        for a in cases {
            for b in cases {
                let av: [[u64; LIMBS]; LANES] = [to_ifma(&a); LANES];
                let bv: [[u64; LIMBS]; LANES] = [to_ifma(&b); LANES];
                let (sum, diff) = unsafe {
                    let (x, y) = (Fq8::load(&av), Fq8::load(&bv));
                    (add_x8(&x, &y).store(), sub_x8(&x, &y).store())
                };
                assert_eq!(from_ifma(&sum[0]), a + b, "{a} + {b}");
                assert_eq!(from_ifma(&diff[0]), a - b, "{a} - {b}");
            }
        }
    }

    /// Reports throughput against the scalar path. Run with:
    /// `cargo test -p snarkvm-algorithms --release ifma::tests::bench --
    /// --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_mul_x8_vs_scalar() {
        use std::time::Instant;
        if skip_if_unsupported() {
            return;
        }
        let mut rng = TestRng::default();
        let a: [Fq; LANES] = std::array::from_fn(|_| Fq::rand(&mut rng));
        let b = Fq::rand(&mut rng);
        let inner = 20_000u32;

        let mut best_scalar = f64::MAX;
        for _ in 0..7 {
            let mut acc = a;
            let t = Instant::now();
            for _ in 0..inner {
                for x in acc.iter_mut() {
                    *x *= b;
                }
            }
            std::hint::black_box(acc);
            best_scalar = best_scalar.min(t.elapsed().as_nanos() as f64 / (inner as f64 * LANES as f64));
        }

        let av: [[u64; LIMBS]; LANES] = std::array::from_fn(|i| to_ifma(&a[i]));
        let bv: [[u64; LIMBS]; LANES] = [to_ifma(&b); LANES];
        let mut best_vec = f64::MAX;
        for _ in 0..7 {
            unsafe {
                let mut x = Fq8::load(&av);
                let y = Fq8::load(&bv);
                let t = Instant::now();
                for _ in 0..inner {
                    x = mul_x8(&x, &y);
                }
                std::hint::black_box(x.store());
                best_vec = best_vec.min(t.elapsed().as_nanos() as f64 / (inner as f64 * LANES as f64));
            }
        }
        println!("  scalar Fq mul (8 independent) : {best_scalar:>6.2} ns/mul");
        println!("  8-lane IFMA mul_x8            : {best_vec:>6.2} ns/mul");
        println!("  speedup                       : {:>6.2}x", best_scalar / best_vec);
    }

    #[test]
    fn test_mul_x8_lanes_are_independent() {
        if skip_if_unsupported() {
            return;
        }
        let mut rng = TestRng::default();
        let a: [Fq; LANES] = std::array::from_fn(|_| Fq::rand(&mut rng));
        let b: [Fq; LANES] = std::array::from_fn(|_| Fq::rand(&mut rng));
        let av: [[u64; LIMBS]; LANES] = std::array::from_fn(|i| to_ifma(&a[i]));
        let bv: [[u64; LIMBS]; LANES] = std::array::from_fn(|i| to_ifma(&b[i]));
        let got = unsafe { mul_x8(&Fq8::load(&av), &Fq8::load(&bv)).store() };
        // Each lane must equal the product of that lane's own inputs only.
        for lane in 0..LANES {
            assert_eq!(from_ifma(&got[lane]), a[lane] * b[lane]);
            for (other, b_other) in b.iter().enumerate() {
                if other != lane {
                    assert_ne!(from_ifma(&got[lane]), a[lane] * b_other, "lanes crossed");
                }
            }
        }
    }
}
