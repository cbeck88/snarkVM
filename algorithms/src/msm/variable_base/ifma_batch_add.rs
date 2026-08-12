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

//! Eight-lane batch affine addition for BLS12-377 G1, on top of the IFMA
//! backend in [`super::ifma`].
//!
//! The scalar version in `batched.rs` threads a single `inversion_tmp` running
//! product through the whole batch, which is inherently serial. This version
//! keeps eight independent chains -- lane `L` owns every pair at index
//! `8k + L` -- and recombines them into one inversion at the end. Batch
//! inversion is exact, so splitting the chain changes only the grouping of the
//! multiplications, not the field elements produced.
//!
//! Points are held in the IFMA Montgomery domain for the lifetime of a
//! [`PointBuf`], so the domain conversion is paid once per MSM rather than once
//! per addition.
//!
//! Against the scalar path as shipped, on a Xeon w9-3475X at n = 2^18, single
//! threaded, both measured in the same run: 1.799s to 1.350s, or 1.33x.
//!
//! Two things contribute. In a separate experiment with both paths forced to
//! the same batch size, the vectorized arithmetic alone was worth 1.11x to
//! 1.15x. The rest comes from this path
//! using a larger batch, which spreads the one field inversion each call pays
//! over more additions. The scalar path is just as sensitive to batch size and
//! would gain similarly if retuned, so that half of the improvement is
//! available to it independently of any of this. Retuning it is deliberately
//! left out of this change: the shipped constant was measured across Intel,
//! AMD and M1, and one machine is not grounds to move it.
//!
//! That is well short of what the kernel manages on its own -- 138.9 ns/pair
//! against 270.0 for the scalar version, once the domain conversion is
//! excluded. The remainder goes to two places. Every gather transposes eight
//! points from array-of-structs into vector lanes and every scatter transposes
//! back, and a point costs 128 bytes here (two coordinates x eight 52-bit limbs
//! held in 64-bit lanes) against 96 for a `G1Affine`, so the kernel moves a
//! third more memory than the scalar one for the same work. Closing either
//! would need a limb-major layout, which the scattered index lists make
//! awkward.

#![allow(unsafe_code)]

use super::{
    batched::BucketPosition,
    ifma::{self, Fq8, LANES, LIMBS, add_ref, from_ifma, mont_mul_ref, sub_ref, to_ifma},
};
use snarkvm_curves::bls12_377::{Fq, G1Affine};
use snarkvm_fields::{Field, One, Zero};

/// A batch of affine points held in the IFMA Montgomery domain.
#[derive(Default)]
pub struct PointBuf {
    x: Vec<[u64; LIMBS]>,
    y: Vec<[u64; LIMBS]>,
    infinity: Vec<bool>,
}

impl PointBuf {
    /// Converts affine points into the IFMA domain. The domain shift is done
    /// in bulk, since converting element-by-element costs more than the
    /// arithmetic it feeds.
    pub fn from_affine(points: &[G1Affine]) -> Self {
        let mut buf = Self {
            x: points.iter().map(|p| ifma::regroup_raw(&p.x)).collect(),
            y: points.iter().map(|p| ifma::regroup_raw(&p.y)).collect(),
            infinity: points.iter().map(|p| p.infinity).collect(),
        };
        ifma::shift_into_domain_slice(&mut buf.x);
        ifma::shift_into_domain_slice(&mut buf.y);
        buf
    }

    /// Creates an empty buffer with room for `capacity` points.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            infinity: Vec::with_capacity(capacity),
        }
    }

    /// Appends one affine point, converting it into the IFMA domain.
    pub fn push(&mut self, p: &G1Affine) {
        self.x.push(to_ifma(&p.x));
        self.y.push(to_ifma(&p.y));
        self.infinity.push(p.infinity);
    }

    /// Copies the point at `src` of `other` onto the end of this buffer.
    pub fn push_from_other(&mut self, other: &PointBuf, src: usize) {
        self.x.push(other.x[src]);
        self.y.push(other.y[src]);
        self.infinity.push(other.infinity[src]);
    }

    /// Overwrites the point at `dst` with `other`'s point at `src`.
    pub fn copy_from(&mut self, dst: usize, other: &PointBuf, src: usize) {
        self.x[dst] = other.x[src];
        self.y[dst] = other.y[src];
        self.infinity[dst] = other.infinity[src];
    }

    /// Converts the point at `i` back to affine form. The coordinates are
    /// preserved even when the infinity flag is set, so that the result is
    /// bit-identical to the scalar path rather than merely equivalent.
    pub fn get(&self, i: usize) -> G1Affine {
        G1Affine::new(from_ifma(&self.x[i]), from_ifma(&self.y[i]), self.infinity[i])
    }

    pub fn len(&self) -> usize {
        self.infinity.len()
    }

    pub fn is_empty(&self) -> bool {
        self.infinity.is_empty()
    }

    pub fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
        self.infinity.clear();
    }
}

/// The running products for the eight inversion chains.
struct Chains {
    tmp: [[u64; LIMBS]; LANES],
}

impl Chains {
    fn new() -> Self {
        Self { tmp: [ifma::ONE; LANES] }
    }

    /// Folds the eight chain products into a single inversion, then hands each
    /// lane back the inverse of its own product.
    ///
    /// The fold runs in the native field: it is only ~24 multiplies, but an
    /// 8-limb multiply in this domain costs far more than a native one, and the
    /// two domain shifts are a single vector operation each.
    fn invert(&self) -> [[u64; LIMBS]; LANES] {
        let t = ifma::from_ifma_x8(&self.tmp);

        // prefix[l] is the product of every chain strictly before l.
        let mut prefix = [Fq::one(); LANES];
        for l in 1..LANES {
            prefix[l] = prefix[l - 1] * t[l - 1];
        }
        // One inversion for the whole batch; the chains hold products of
        // nonzero denominators, so this cannot fail.
        let inv_total = (prefix[LANES - 1] * t[LANES - 1]).inverse().expect("batch denominator is nonzero");

        let mut out = [Fq::one(); LANES];
        let mut suffix = Fq::one();
        for l in (0..LANES).rev() {
            out[l] = inv_total * prefix[l] * suffix;
            suffix *= t[l];
        }
        ifma::to_ifma_x8(&out)
    }
}

/// For each `(i, j)` in `index`, sets point `i` to `point i + point j`. The
/// state of point `j` becomes unspecified, matching the scalar version.
pub fn batch_add_in_place(buf: &mut PointBuf, index: &[(u32, u32)]) {
    let mut scratch = Scratch::with_capacity(index.len());
    batch_add_in_place_with(buf, index, &mut scratch);
}

/// Reusable index buffers, so `pair_add` does not allocate per call.
#[derive(Default)]
pub struct Scratch {
    a_idx: Vec<usize>,
    b_idx: Vec<usize>,
    /// Staged `b` operands for the write path. The doubling branch of the
    /// first pass mutates `b`, and the second pass reads that mutation, so the
    /// operands cannot simply be re-read from the source buffer.
    b_buf: PointBuf,
}

impl Scratch {
    /// Creates scratch sized for a batch of `capacity` pairs.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            a_idx: Vec::with_capacity(capacity),
            b_idx: Vec::with_capacity(capacity),
            b_buf: PointBuf::with_capacity(capacity),
        }
    }
}

/// As [`batch_add_in_place`], reusing the caller's scratch.
pub fn batch_add_in_place_with(buf: &mut PointBuf, index: &[(u32, u32)], scratch: &mut Scratch) {
    scratch.a_idx.clear();
    scratch.b_idx.clear();
    scratch.a_idx.extend(index.iter().map(|(i, _)| *i as usize));
    scratch.b_idx.extend(index.iter().map(|(_, j)| *j as usize));
    let Scratch { a_idx, b_idx, .. } = scratch;
    pair_add(buf, a_idx, None, b_idx);
}

/// Adds `b[b_idx[k]]` into `a[a_idx[k]]` for every `k`, over one batch
/// inversion split across eight lanes.
///
/// When `b_src` is `None` the operands live in `a` itself, which is what the
/// in-place phase needs. The generic path never writes to a `b` operand, so the
/// eight `b` values are read out before `a` is touched.
fn pair_add(a: &mut PointBuf, a_idx: &[usize], mut b_src: Option<&mut PointBuf>, b_idx: &[usize]) {
    debug_assert_eq!(a_idx.len(), b_idx.len());
    if a_idx.is_empty() {
        return;
    }
    debug_assert!(ifma::is_available());

    let half = ifma::HALF;
    let mut chains = Chains::new();

    let n = a_idx.len();
    let groups = n.div_ceil(LANES);

    for g in 0..groups {
        let lo = g * LANES;
        let hi = (lo + LANES).min(n);
        let generic = hi - lo == LANES && all_generic(a, a_idx, b_src.as_deref(), b_idx, lo);
        if generic {
            // SAFETY: `is_available` was asserted above.
            unsafe { loop_1_x8(a, &a_idx[lo..hi], b_src.as_deref(), &b_idx[lo..hi], &mut chains) };
        } else {
            for (lane, k) in (lo..hi).enumerate() {
                loop_1_scalar(a, a_idx[k], b_src.as_deref_mut(), b_idx[k], lane, &mut chains, &half);
            }
        }
    }

    let mut acc = chains.invert();

    for g in (0..groups).rev() {
        let lo = g * LANES;
        let hi = (lo + LANES).min(n);
        let generic = hi - lo == LANES && all_generic(a, a_idx, b_src.as_deref(), b_idx, lo);
        if generic {
            // SAFETY: as above.
            unsafe { loop_2_x8(a, &a_idx[lo..hi], b_src.as_deref(), &b_idx[lo..hi], &mut acc) };
        } else {
            for (lane, k) in (lo..hi).enumerate() {
                loop_2_scalar(a, a_idx[k], b_src.as_deref(), b_idx[k], lane, &mut acc);
            }
        }
    }
}

/// True when all eight pairs starting at `lo` take the generic chord path.
#[inline]
fn all_generic(a: &PointBuf, a_idx: &[usize], b_src: Option<&PointBuf>, b_idx: &[usize], lo: usize) -> bool {
    let b = b_src.unwrap_or(a);
    (lo..lo + LANES).all(|k| !a.infinity[a_idx[k]] && !b.infinity[b_idx[k]] && a.x[a_idx[k]] != b.x[b_idx[k]])
}

/// Gathers eight points' coordinates into vector registers.
///
/// # Safety
/// Requires AVX-512F.
#[target_feature(enable = "avx512f")]
unsafe fn gather(src: &[[u64; LIMBS]], idx: [usize; LANES]) -> Fq8 {
    let staged: [[u64; LIMBS]; LANES] = std::array::from_fn(|l| src[idx[l]]);
    unsafe { Fq8::load(&staged) }
}

/// Vectorized first pass over eight generic pairs.
///
/// # Safety
/// Requires AVX-512F and AVX512-IFMA.
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn loop_1_x8(a: &mut PointBuf, a_idx: &[usize], b_src: Option<&PointBuf>, b_idx: &[usize], chains: &mut Chains) {
    unsafe {
        let ia: [usize; LANES] = std::array::from_fn(|l| a_idx[l]);
        let ib: [usize; LANES] = std::array::from_fn(|l| b_idx[l]);

        // Read the b operands out first; the generic path never writes them, so
        // this is sound even when they alias `a`.
        let b = b_src.unwrap_or(&*a);
        let bx = gather(&b.x, ib);
        let by = gather(&b.y, ib);
        let ax = gather(&a.x, ia);
        let ay = gather(&a.y, ia);
        let tmp = Fq8::load(&chains.tmp);

        // denominator = x1 - x2, numerator = y1 - y2.
        let den = ifma::sub_x8(&ax, &bx);
        let num = ifma::sub_x8(&ay, &by);
        // a.y holds numerator * running product; the chain then absorbs the
        // denominator.
        let new_ay = ifma::mul_x8(&num, &tmp);
        let new_tmp = ifma::mul_x8(&tmp, &den);

        let sx = den.store();
        let sy = new_ay.store();
        for l in 0..LANES {
            a.x[ia[l]] = sx[l];
            a.y[ia[l]] = sy[l];
        }
        chains.tmp = new_tmp.store();
    }
}

/// Vectorized second pass over eight generic pairs.
///
/// # Safety
/// Requires AVX-512F and AVX512-IFMA.
#[target_feature(enable = "avx512f,avx512ifma")]
unsafe fn loop_2_x8(
    a: &mut PointBuf,
    a_idx: &[usize],
    b_src: Option<&PointBuf>,
    b_idx: &[usize],
    acc: &mut [[u64; LIMBS]; LANES],
) {
    unsafe {
        let ia: [usize; LANES] = std::array::from_fn(|l| a_idx[l]);
        let ib: [usize; LANES] = std::array::from_fn(|l| b_idx[l]);

        let b = b_src.unwrap_or(&*a);
        let bx = gather(&b.x, ib);
        let by = gather(&b.y, ib);
        let ax = gather(&a.x, ia);
        let ay = gather(&a.y, ia);
        let inv = Fq8::load(acc);

        let lambda = ifma::mul_x8(&ay, &inv);
        // Peel this layer off the running inverse.
        let new_inv = ifma::mul_x8(&inv, &ax);

        // a.x currently holds x1 - x2, so adding 2*x2 recovers x1 + x2.
        let sum_x = ifma::add_x8(&ax, &ifma::add_x8(&bx, &bx));
        let x3 = ifma::sub_x8(&ifma::mul_x8(&lambda, &lambda), &sum_x);
        let y3 = ifma::sub_x8(&ifma::mul_x8(&lambda, &ifma::sub_x8(&bx, &x3)), &by);

        let sx = x3.store();
        let sy = y3.store();
        for l in 0..LANES {
            a.x[ia[l]] = sx[l];
            a.y[ia[l]] = sy[l];
        }
        *acc = new_inv.store();
    }
}

/// Scalar first pass for one pair, kept in the IFMA domain so that degenerate
/// lanes need no domain switch. Mirrors `batch_add_loop_1`.
fn loop_1_scalar(
    a: &mut PointBuf,
    ai: usize,
    mut b_src: Option<&mut PointBuf>,
    bi: usize,
    lane: usize,
    chains: &mut Chains,
    half: &[u64; LIMBS],
) {
    let (bx, by, binf) = match &b_src {
        Some(b) => (b.x[bi], b.y[bi], b.infinity[bi]),
        None => (a.x[bi], a.y[bi], a.infinity[bi]),
    };
    if a.infinity[ai] || binf {
        return;
    }
    if a.x[ai] == bx {
        if a.y[ai] == by {
            // Doubling. WEIERSTRASS_A is zero for BLS12-377 G1, so the
            // numerator is 3x^2.
            let x_sq = mont_mul_ref(&bx, &bx);
            let num = add_ref(&add_ref(&x_sq, &x_sq), &x_sq);
            let new_bx = sub_ref(&bx, &by);
            let new_by = sub_ref(&by, &mont_mul_ref(&num, half));
            a.x[ai] = add_ref(&by, &by);
            a.y[ai] = mont_mul_ref(&num, &chains.tmp[lane]);
            chains.tmp[lane] = mont_mul_ref(&chains.tmp[lane], &a.x[ai]);
            // The second pass reads these back, so they must land wherever the
            // b operand actually lives.
            match &mut b_src {
                Some(b) => {
                    b.x[bi] = new_bx;
                    b.y[bi] = new_by;
                }
                None => {
                    a.x[bi] = new_bx;
                    a.y[bi] = new_by;
                }
            }
        } else {
            a.infinity[ai] = true;
            match &mut b_src {
                Some(b) => b.infinity[bi] = true,
                None => a.infinity[bi] = true,
            }
        }
        return;
    }
    let den = sub_ref(&a.x[ai], &bx);
    let num = sub_ref(&a.y[ai], &by);
    a.x[ai] = den;
    a.y[ai] = mont_mul_ref(&num, &chains.tmp[lane]);
    chains.tmp[lane] = mont_mul_ref(&chains.tmp[lane], &den);
}

/// Scalar second pass for one pair. Mirrors `batch_add_loop_2`.
fn loop_2_scalar(
    a: &mut PointBuf,
    ai: usize,
    b_src: Option<&PointBuf>,
    bi: usize,
    lane: usize,
    acc: &mut [[u64; LIMBS]; LANES],
) {
    let (bx, by, binf) = match b_src {
        Some(b) => (b.x[bi], b.y[bi], b.infinity[bi]),
        None => (a.x[bi], a.y[bi], a.infinity[bi]),
    };
    if a.infinity[ai] {
        a.x[ai] = bx;
        a.y[ai] = by;
        a.infinity[ai] = binf;
        return;
    }
    if binf {
        return;
    }
    let lambda = mont_mul_ref(&a.y[ai], &acc[lane]);
    acc[lane] = mont_mul_ref(&acc[lane], &a.x[ai]);

    let sum_x = add_ref(&a.x[ai], &add_ref(&bx, &bx));
    let x3 = sub_ref(&mont_mul_ref(&lambda, &lambda), &sum_x);
    let y3 = sub_ref(&mont_mul_ref(&lambda, &sub_ref(&bx, &x3)), &by);
    a.x[ai] = x3;
    a.y[ai] = y3;
}

/// Mirrors `batched::batch_add_write`: for each `(i, j)` writes `bases[i] +
/// bases[j]` (or just `bases[i]` when `j` is the `!0` sentinel) onto the end of
/// `out`.
fn batch_add_write(bases: &PointBuf, index: &[(u32, u32)], out: &mut PointBuf, scratch: &mut Scratch) {
    scratch.a_idx.clear();
    scratch.b_idx.clear();
    scratch.b_buf.clear();
    for (idx, idy) in index.iter() {
        out.push_from_other(bases, *idx as usize);
        if *idy != !0u32 {
            scratch.a_idx.push(out.len() - 1);
            scratch.b_idx.push(scratch.b_buf.len());
            scratch.b_buf.push_from_other(bases, *idy as usize);
        }
    }
    let Scratch { a_idx, b_idx, b_buf } = scratch;
    pair_add(out, a_idx, Some(b_buf), b_idx);
}

/// Batch size for the vectorized path, overridable via
/// `SNARKVM_IFMA_BATCH_SIZE`.
///
/// This is a tuning knob, not a correctness one: it only sets how many
/// additions share a single field inversion.
///
/// The default rests on one sweep, at n = 2^18, single threaded, on a Xeon
/// w9-3475X, with the shipped scalar path measured alongside for reference.
/// Absolute times move with machine state; the shape, and the value it picks
/// out, held across re-runs:
///
/// ```text
/// batch size    256     512    1024    2048    4096    8192
/// ifma        2.009   1.720   1.604   1.570   1.543   1.512
/// scalar         --   1.942   1.849   1.764   1.714      --
///
/// scalar, as shipped (batch size 300): 2.043
/// ```
///
/// Note that the scalar path has the same shape, so the sensitivity is not an
/// artifact of the wider representation used here: both pay one field
/// inversion per call, which only amortizes with batch size.
///
/// 4096 is picked because the gain past it is under 2% while each pair touches
/// two 128-byte points, which keeps the working set near 1 MiB rather than
/// running off the end of L2. That basis is much narrower than the one behind
/// `batched::batch_size`, which was tuned across Intel, AMD and M1, so the
/// value is left adjustable until it has been measured on more hardware.
fn ifma_batch_size() -> usize {
    const DEFAULT: usize = 4096;
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("SNARKVM_IFMA_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT)
    })
}

/// Mirrors `batched::batch_add` for BLS12-377 G1.
pub fn batch_add(num_buckets: usize, bases_ifma: &PointBuf, bucket_positions: &mut [BucketPosition]) -> Vec<G1Affine> {
    assert!(bases_ifma.len() >= bucket_positions.len());
    assert!(!bases_ifma.is_empty());

    let batch_size = ifma_batch_size();
    bucket_positions.sort_unstable();

    let mut num_scalars = bucket_positions.len();
    let mut all_ones = true;
    let mut new_scalar_length = 0;
    let mut global_counter = 0;
    let mut local_counter = 1;
    let mut number_of_bases_in_batch = 0;

    let mut instr = Vec::<(u32, u32)>::with_capacity(batch_size);
    let mut new_bases = PointBuf::with_capacity(bases_ifma.len());
    let mut scratch = Scratch::with_capacity(batch_size);

    while global_counter < num_scalars {
        let current_bucket = bucket_positions[global_counter].bucket_index;
        while global_counter + 1 < num_scalars && bucket_positions[global_counter + 1].bucket_index == current_bucket {
            global_counter += 1;
            local_counter += 1;
        }
        if current_bucket >= num_buckets as u32 {
            local_counter = 1;
        } else if local_counter > 1 {
            if local_counter > 2 {
                all_ones = false;
            }
            let is_odd = local_counter % 2 == 1;
            let half = local_counter / 2;
            for i in 0..half {
                instr.push((
                    bucket_positions[global_counter - (local_counter - 1) + 2 * i].scalar_index,
                    bucket_positions[global_counter - (local_counter - 1) + 2 * i + 1].scalar_index,
                ));
                bucket_positions[new_scalar_length + i] =
                    BucketPosition { bucket_index: current_bucket, scalar_index: (new_scalar_length + i) as u32 };
            }
            if is_odd {
                instr.push((bucket_positions[global_counter].scalar_index, !0u32));
                bucket_positions[new_scalar_length + half] =
                    BucketPosition { bucket_index: current_bucket, scalar_index: (new_scalar_length + half) as u32 };
            }
            new_scalar_length += half + (local_counter % 2);
            number_of_bases_in_batch += half;
            local_counter = 1;

            if number_of_bases_in_batch >= batch_size / 2 {
                batch_add_write(bases_ifma, &instr, &mut new_bases, &mut scratch);
                instr.clear();
                number_of_bases_in_batch = 0;
            }
        } else {
            instr.push((bucket_positions[global_counter].scalar_index, !0u32));
            bucket_positions[new_scalar_length] =
                BucketPosition { bucket_index: current_bucket, scalar_index: new_scalar_length as u32 };
            new_scalar_length += 1;
        }
        global_counter += 1;
    }
    if !instr.is_empty() {
        batch_add_write(bases_ifma, &instr, &mut new_bases, &mut scratch);
        instr.clear();
    }
    global_counter = 0;
    number_of_bases_in_batch = 0;
    local_counter = 1;
    num_scalars = new_scalar_length;
    new_scalar_length = 0;

    while !all_ones {
        all_ones = true;
        while global_counter < num_scalars {
            let current_bucket = bucket_positions[global_counter].bucket_index;
            while global_counter + 1 < num_scalars
                && bucket_positions[global_counter + 1].bucket_index == current_bucket
            {
                global_counter += 1;
                local_counter += 1;
            }
            if current_bucket >= num_buckets as u32 {
                local_counter = 1;
            } else if local_counter > 1 {
                if local_counter != 2 {
                    all_ones = false;
                }
                let is_odd = local_counter % 2 == 1;
                let half = local_counter / 2;
                for i in 0..half {
                    instr.push((
                        bucket_positions[global_counter - (local_counter - 1) + 2 * i].scalar_index,
                        bucket_positions[global_counter - (local_counter - 1) + 2 * i + 1].scalar_index,
                    ));
                    bucket_positions[new_scalar_length + i] =
                        bucket_positions[global_counter - (local_counter - 1) + 2 * i];
                }
                if is_odd {
                    bucket_positions[new_scalar_length + half] = bucket_positions[global_counter];
                }
                new_scalar_length += half + (local_counter % 2);
                number_of_bases_in_batch += half;
                local_counter = 1;

                if number_of_bases_in_batch >= batch_size / 2 {
                    batch_add_in_place_with(&mut new_bases, &instr, &mut scratch);
                    instr.clear();
                    number_of_bases_in_batch = 0;
                }
            } else {
                bucket_positions[new_scalar_length] = bucket_positions[global_counter];
                new_scalar_length += 1;
            }
            global_counter += 1;
        }
        if !instr.is_empty() {
            batch_add_in_place_with(&mut new_bases, &instr, &mut scratch);
            instr.clear();
        }
        global_counter = 0;
        number_of_bases_in_batch = 0;
        local_counter = 1;
        num_scalars = new_scalar_length;
        new_scalar_length = 0;
    }

    let mut res = vec![G1Affine::zero(); num_buckets];
    for bucket_position in bucket_positions.iter().take(num_scalars) {
        res[bucket_position.bucket_index as usize] = new_bases.get(bucket_position.scalar_index as usize);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkvm_curves::{AffineCurve, ProjectiveCurve, bls12_377::Fr};
    use snarkvm_utilities::{TestRng, Uniform};

    /// The scalar batch addition from `batched.rs`, reproduced so the
    /// vectorized path can be compared against it directly.
    fn scalar_batch_add(bases: &mut [G1Affine], index: &[(u32, u32)]) {
        let mut inversion_tmp = Fq::one();
        let half = Fq::half();
        for (idx, idy) in index.iter() {
            let (a, b) = if idx < idy {
                let (x, y) = bases.split_at_mut(*idy as usize);
                (&mut x[*idx as usize], &mut y[0])
            } else {
                let (x, y) = bases.split_at_mut(*idx as usize);
                (&mut y[0], &mut x[*idy as usize])
            };
            G1Affine::batch_add_loop_1(a, b, &half, &mut inversion_tmp);
        }
        inversion_tmp = inversion_tmp.inverse().unwrap();
        for (idx, idy) in index.iter().rev() {
            let (a, b) = if idx < idy {
                let (x, y) = bases.split_at_mut(*idy as usize);
                (&mut x[*idx as usize], y[0])
            } else {
                let (x, y) = bases.split_at_mut(*idx as usize);
                (&mut y[0], x[*idy as usize])
            };
            G1Affine::batch_add_loop_2(a, b, &mut inversion_tmp);
        }
    }

    fn run_case(n_pairs: usize, rng: &mut TestRng, mutate: impl Fn(&mut Vec<G1Affine>, &mut Vec<(u32, u32)>)) {
        if !ifma::is_available() {
            eprintln!("skipping: CPU lacks AVX512-IFMA");
            return;
        }
        let mut points: Vec<G1Affine> = (0..2 * n_pairs).map(|_| G1Affine::rand(rng)).collect();
        let mut index: Vec<(u32, u32)> = (0..n_pairs).map(|k| (2 * k as u32, 2 * k as u32 + 1)).collect();
        mutate(&mut points, &mut index);

        let mut expected = points.clone();
        scalar_batch_add(&mut expected, &index);

        let mut buf = PointBuf::from_affine(&points);
        batch_add_in_place(&mut buf, &index);

        for (k, (idx, _)) in index.iter().enumerate() {
            let got = buf.get(*idx as usize);
            assert_eq!(got, expected[*idx as usize], "pair {k} (point {idx})");
        }
    }

    #[test]
    fn test_matches_scalar_generic() {
        let mut rng = TestRng::default();
        for n in [1usize, 7, 8, 9, 16, 33, 100] {
            run_case(n, &mut rng, |_, _| {});
        }
    }

    #[test]
    fn test_matches_scalar_with_infinities() {
        let mut rng = TestRng::default();
        run_case(64, &mut rng, |points, _| {
            points[0] = G1Affine::zero();
            points[3] = G1Affine::zero();
            points[20] = G1Affine::zero();
            points[21] = G1Affine::zero();
        });
    }

    #[test]
    fn test_matches_scalar_with_doublings() {
        let mut rng = TestRng::default();
        run_case(64, &mut rng, |points, _| {
            // Equal points force the doubling branch.
            points[1] = points[0];
            points[11] = points[10];
        });
    }

    #[test]
    fn test_matches_scalar_with_negations() {
        let mut rng = TestRng::default();
        run_case(64, &mut rng, |points, _| {
            // P + (-P) sends both operands to infinity.
            points[5] = -points[4];
            points[41] = -points[40];
        });
    }

    /// Isolates the kernel from domain conversion. Run with:
    /// `cargo test -p snarkvm-algorithms --release ifma_batch_add::tests::bench
    /// -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_kernel_vs_scalar() {
        use std::time::Instant;
        if !ifma::is_available() {
            return;
        }
        let mut rng = TestRng::default();
        let n_pairs = 4096;
        let points: Vec<G1Affine> = (0..2 * n_pairs).map(|_| G1Affine::rand(&mut rng)).collect();
        let index: Vec<(u32, u32)> = (0..n_pairs).map(|k| (2 * k as u32, 2 * k as u32 + 1)).collect();

        let mut best_scalar = f64::MAX;
        for _ in 0..7 {
            let mut pts = points.clone();
            let t = Instant::now();
            scalar_batch_add(&mut pts, &index);
            std::hint::black_box(&pts);
            best_scalar = best_scalar.min(t.elapsed().as_nanos() as f64 / n_pairs as f64);
        }

        // Conversion excluded: the buffer is built once, outside the timer.
        let mut best_kernel = f64::MAX;
        for _ in 0..7 {
            let mut buf = PointBuf::from_affine(&points);
            let t = Instant::now();
            batch_add_in_place(&mut buf, &index);
            std::hint::black_box(buf.len());
            best_kernel = best_kernel.min(t.elapsed().as_nanos() as f64 / n_pairs as f64);
        }

        let mut best_conv = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            std::hint::black_box(PointBuf::from_affine(&points));
            best_conv = best_conv.min(t.elapsed().as_nanos() as f64 / n_pairs as f64);
        }

        println!("  scalar batch add        : {best_scalar:>7.1} ns/pair");
        println!("  ifma kernel (no convert): {best_kernel:>7.1} ns/pair  ({:.2}x)", best_scalar / best_kernel);
        println!("  domain conversion       : {best_conv:>7.1} ns/pair (2 points each)");
        println!(
            "  ifma incl. conversion   : {:>7.1} ns/pair  ({:.2}x)",
            best_kernel + best_conv,
            best_scalar / (best_kernel + best_conv)
        );
    }

    /// Splits the wired path into kernel vs. staging, to see which dominates.
    #[test]
    #[ignore]
    fn bench_write_path() {
        use std::time::Instant;
        if !ifma::is_available() {
            return;
        }
        let mut rng = TestRng::default();
        let n_pairs = 150; // matches `batch_size / 2` for a small MSM
        let reps = 2000;
        let points: Vec<G1Affine> = (0..2 * n_pairs).map(|_| G1Affine::rand(&mut rng)).collect();
        let index: Vec<(u32, u32)> = (0..n_pairs).map(|k| (2 * k as u32, 2 * k as u32 + 1)).collect();
        let bases = PointBuf::from_affine(&points);

        let mut best_scalar = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            for _ in 0..reps {
                let mut pts = points.clone();
                scalar_batch_add(&mut pts, &index);
                std::hint::black_box(&pts);
            }
            best_scalar = best_scalar.min(t.elapsed().as_nanos() as f64 / (reps * n_pairs) as f64);
        }
        // Subtract the clone the scalar loop pays, so only the addition is timed.
        let mut clone_cost = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(points.clone());
            }
            clone_cost = clone_cost.min(t.elapsed().as_nanos() as f64 / (reps * n_pairs) as f64);
        }

        let mut best_write = f64::MAX;
        for _ in 0..7 {
            let mut out = PointBuf::with_capacity(n_pairs);
            let mut scratch = Scratch::with_capacity(n_pairs);
            let t = Instant::now();
            for _ in 0..reps {
                out.clear();
                batch_add_write(&bases, &index, &mut out, &mut scratch);
                std::hint::black_box(out.len());
            }
            best_write = best_write.min(t.elapsed().as_nanos() as f64 / (reps * n_pairs) as f64);
        }

        println!("  scalar batch add (incl clone) : {best_scalar:>7.1} ns/pair");
        println!("  clone alone                   : {clone_cost:>7.1} ns/pair");
        println!("  scalar batch add (net)        : {:>7.1} ns/pair", best_scalar - clone_cost);
        println!(
            "  ifma batch_add_write          : {best_write:>7.1} ns/pair  ({:.2}x)",
            (best_scalar - clone_cost) / best_write
        );
    }

    /// Isolates the fixed per-call cost of `pair_add` from its per-pair cost.
    #[test]
    #[ignore]
    fn bench_per_call_overhead() {
        use std::time::Instant;
        if !ifma::is_available() {
            return;
        }
        let mut rng = TestRng::default();
        let points: Vec<G1Affine> = (0..40_000).map(|_| G1Affine::rand(&mut rng)).collect();
        let buf0 = PointBuf::from_affine(&points);
        println!("  pairs/call   ns/pair");
        let mut prev = (0f64, 0f64);
        for pairs in [8usize, 32, 128, 512, 2048, 8192] {
            let index: Vec<(u32, u32)> = (0..pairs).map(|k| (2 * k as u32, 2 * k as u32 + 1)).collect();
            let reps = (65_536 / pairs).max(1);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let mut buf = PointBuf::from_affine(&points[..2 * pairs]);
                let t = Instant::now();
                for _ in 0..reps {
                    batch_add_in_place(&mut buf, &index);
                }
                std::hint::black_box(buf.len());
                best = best.min(t.elapsed().as_nanos() as f64 / (reps * pairs) as f64);
            }
            println!("  {pairs:>8}   {best:>7.1}");
            if prev.0 > 0.0 {
                // Two points on ns/pair = per_pair + fixed/pairs give the split.
                let fixed = (prev.1 - best) / (1.0 / prev.0 - 1.0 / pairs as f64);
                println!("           -> implied fixed cost per call: {:.0} ns", fixed);
            }
            prev = (pairs as f64, best);
        }
        let _ = buf0.len();
    }

    /// A frozen test vector for the write path.
    ///
    /// Bases are given as multiples of the prime-order generator, so both the
    /// inputs and the expected outputs stay readable and can be checked by
    /// hand: `[i]G + [j]G` must be `[i + j]G`. `0` denotes the point at
    /// infinity, and a negative entry denotes `-[k]G`.
    struct Vector {
        /// The edge case this vector pins down.
        what: &'static str,
        /// Bases, as generator multiples.
        bases: &'static [i64],
        /// Index pairs into `bases`. A second element of `NOOP` is the `!0`
        /// sentinel the caller uses for a bucket holding an odd count.
        index: &'static [(u32, u32)],
        /// Expected result per index entry, as a generator multiple. Produced
        /// by running the scalar algorithm; see `regenerate_test_vectors`.
        expect: &'static [i64],
    }

    const NOOP: u32 = !0u32;

    /// Concrete cases, each pinning one thing the vectorized path has to agree
    /// with the scalar one about.
    const VECTORS: &[Vector] = &[
        Vector {
            what: "a single generic pair: distinct x, the plain chord case",
            bases: &[1, 2],
            index: &[(0, 1)],
            expect: &[3],
        },
        Vector {
            what: "odd bucket count: a trailing NOOP entry is copied, not added",
            bases: &[1, 2, 5],
            index: &[(0, 1), (2, NOOP)],
            expect: &[3, 5],
        },
        Vector {
            what: "duplicate points: takes the doubling branch, which mutates the b operand and must carry that into the second pass",
            bases: &[3, 3],
            index: &[(0, 1)],
            expect: &[6],
        },
        Vector {
            what: "a at infinity: the second pass copies b over a",
            bases: &[0, 7],
            index: &[(0, 1)],
            expect: &[7],
        },
        Vector { what: "b at infinity: a passes through untouched", bases: &[7, 0], index: &[(0, 1)], expect: &[7] },
        Vector { what: "both at infinity: stays at infinity", bases: &[0, 0], index: &[(0, 1)], expect: &[0] },
        Vector {
            what: "P + (-P): equal x, opposite y, so both operands go to infinity",
            bases: &[9, -9],
            index: &[(0, 1)],
            expect: &[0],
        },
        Vector {
            what: "exactly one full lane group: eight generic pairs, the vectorized path with no scalar tail",
            bases: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            index: &[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
            expect: &[3, 7, 11, 15, 19, 23, 27, 31],
        },
        Vector {
            what: "one full group plus a one-pair tail: exercises the scalar remainder path alongside the vectorized one",
            bases: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            index: &[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)],
            expect: &[3, 7, 11, 15, 19, 23, 27, 31, 35],
        },
        Vector {
            what: "a doubling inside an otherwise generic group: one degenerate lane sends the whole group down the scalar path",
            bases: &[1, 2, 3, 4, 5, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            index: &[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
            expect: &[3, 7, 10, 15, 19, 23, 27, 31],
        },
        Vector {
            what: "doublings straddling a group boundary: last lane of group 0 and first lane of group 1",
            bases: &[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 6, 6, 8, 8, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32,
            ],
            index: &[
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
                (10, 11),
                (12, 13),
                (14, 15),
                (16, 17),
                (18, 19),
                (20, 21),
                (22, 23),
                (24, 25),
                (26, 27),
                (28, 29),
                (30, 31),
            ],
            expect: &[3, 7, 11, 15, 19, 23, 27, 12, 16, 39, 43, 47, 51, 55, 59, 63],
        },
        Vector {
            what: "a doubling, an infinity and a negation together in one group",
            bases: &[1, 2, 4, 4, 0, 6, 7, -7, 9, 10, 11, 12, 13, 14, 15, 16],
            index: &[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
            expect: &[3, 8, 6, 0, 19, 23, 27, 31],
        },
        Vector {
            what: "every pair a doubling: no lane ever takes the vector path",
            bases: &[2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
            index: &[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)],
            expect: &[4, 6, 8, 10, 12, 14, 16, 18],
        },
        Vector {
            what: "NOOP entries interleaved through a full group, so the pair list and the output list have different lengths",
            bases: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index: &[(0, 1), (2, NOOP), (3, 4), (5, NOOP), (6, 7), (8, 9)],
            expect: &[3, 3, 9, 6, 15, 19],
        },
    ];

    /// Builds `[k]G`, with `0` meaning the point at infinity.
    fn multiple(k: i64) -> G1Affine {
        if k == 0 {
            return G1Affine::zero();
        }
        let g = G1Affine::prime_subgroup_generator().to_projective();
        let p = (g * Fr::from(k.unsigned_abs())).to_affine();
        if k < 0 { -p } else { p }
    }

    #[test]
    fn test_frozen_vectors_match_scalar_answers() {
        if !ifma::is_available() {
            eprintln!("skipping: CPU lacks AVX512-IFMA");
            return;
        }
        for v in VECTORS {
            let points: Vec<G1Affine> = v.bases.iter().map(|k| multiple(*k)).collect();
            let bases = PointBuf::from_affine(&points);
            let mut out = PointBuf::with_capacity(v.index.len());
            let mut scratch = Scratch::with_capacity(v.index.len());
            batch_add_write(&bases, v.index, &mut out, &mut scratch);

            assert_eq!(out.len(), v.expect.len(), "{}", v.what);
            for (k, want) in v.expect.iter().enumerate() {
                let got = out.get(k);
                let expected = multiple(*want);
                // For a result at infinity only the flag is meaningful: the
                // coordinates left behind are never read, because every
                // consumer short-circuits on `is_zero`.
                if expected.is_zero() {
                    assert!(got.is_zero(), "{}: entry {k} should be infinity, got {got}", v.what);
                } else {
                    assert_eq!(got, expected, "{}: entry {k}", v.what);
                }
            }
        }
    }

    /// Recomputes the `expect` fields from the scalar algorithm and prints
    /// them. Run this if the vectors ever need regenerating:
    /// `cargo test -p snarkvm-algorithms --release regenerate_test_vectors --
    /// --ignored --nocapture`
    #[test]
    #[ignore]
    fn regenerate_test_vectors() {
        // Enough multiples of G to name any sum these vectors can produce.
        let mut table = std::collections::HashMap::new();
        for k in -128i64..=128 {
            table.insert(multiple(k).to_string(), k);
        }
        for v in VECTORS {
            let points: Vec<G1Affine> = v.bases.iter().map(|k| multiple(*k)).collect();
            let got = scalar_batch_add_write(&points, v.index);
            let named: Vec<String> = got
                .iter()
                .map(|p| match table.get(&p.to_string()) {
                    Some(k) => k.to_string(),
                    None if p.is_zero() => "0".to_string(),
                    None => format!("<unknown point {p}>"),
                })
                .collect();
            println!("  // {}", v.what);
            println!("  expect: &[{}],", named.join(", "));
        }
    }

    /// `batched::batch_add_write`, reproduced so the write path can be compared
    /// directly. Note that it saves the *mutated* `b` for the second pass.
    fn scalar_batch_add_write(bases: &[G1Affine], index: &[(u32, u32)]) -> Vec<G1Affine> {
        let mut inversion_tmp = Fq::one();
        let half = Fq::half();
        let mut out: Vec<G1Affine> = Vec::with_capacity(index.len());
        let mut scratch: Vec<Option<G1Affine>> = Vec::with_capacity(index.len());
        for (idx, idy) in index.iter() {
            if *idy == !0u32 {
                out.push(bases[*idx as usize]);
                scratch.push(None);
            } else {
                let (mut a, mut b) = (bases[*idx as usize], bases[*idy as usize]);
                G1Affine::batch_add_loop_1(&mut a, &mut b, &half, &mut inversion_tmp);
                out.push(a);
                scratch.push(Some(b));
            }
        }
        inversion_tmp = inversion_tmp.inverse().unwrap();
        for (a, op_b) in out.iter_mut().rev().zip(scratch.iter().rev()) {
            if let Some(b) = op_b {
                G1Affine::batch_add_loop_2(a, *b, &mut inversion_tmp);
            }
        }
        out
    }

    fn run_write_case(rng: &mut TestRng, mutate: impl Fn(&mut Vec<G1Affine>)) {
        if !ifma::is_available() {
            return;
        }
        let n_pairs = 40usize;
        let mut points: Vec<G1Affine> = (0..2 * n_pairs).map(|_| G1Affine::rand(rng)).collect();
        mutate(&mut points);
        // Interleave a few no-op entries, as the real caller does for buckets
        // holding an odd number of points.
        let mut index: Vec<(u32, u32)> = Vec::new();
        for k in 0..n_pairs {
            if k % 7 == 3 {
                index.push((2 * k as u32, !0u32));
            } else {
                index.push((2 * k as u32, 2 * k as u32 + 1));
            }
        }

        let expected = scalar_batch_add_write(&points, &index);

        let bases = PointBuf::from_affine(&points);
        let mut out = PointBuf::with_capacity(index.len());
        let mut scratch = Scratch::with_capacity(index.len());
        batch_add_write(&bases, &index, &mut out, &mut scratch);

        for (k, want) in expected.iter().enumerate() {
            assert_eq!(out.get(k), *want, "entry {k}");
        }
    }

    #[test]
    fn test_write_path_matches_scalar_generic() {
        let mut rng = TestRng::default();
        run_write_case(&mut rng, |_| {});
    }

    #[test]
    fn test_write_path_matches_scalar_with_doublings() {
        // The doubling branch mutates the b operand and the second pass reads
        // that mutation back, which the write path has to preserve.
        let mut rng = TestRng::default();
        run_write_case(&mut rng, |points| {
            points[1] = points[0];
            points[11] = points[10];
            points[65] = points[64];
        });
    }

    #[test]
    fn test_write_path_matches_scalar_with_infinities_and_negations() {
        let mut rng = TestRng::default();
        run_write_case(&mut rng, |points| {
            points[4] = G1Affine::zero();
            points[9] = G1Affine::zero();
            points[13] = -points[12];
            points[21] = -points[20];
        });
    }

    #[test]
    fn test_sum_is_correct() {
        if !ifma::is_available() {
            return;
        }
        let mut rng = TestRng::default();
        let points: Vec<G1Affine> = (0..64).map(|_| G1Affine::rand(&mut rng)).collect();
        let index: Vec<(u32, u32)> = (0..32).map(|k| (2 * k, 2 * k + 1)).collect();
        let mut buf = PointBuf::from_affine(&points);
        batch_add_in_place(&mut buf, &index);
        for k in 0..32usize {
            let expected = points[2 * k].to_projective() + points[2 * k + 1].to_projective();
            assert_eq!(buf.get(2 * k), expected.to_affine(), "pair {k}");
        }
    }
}
