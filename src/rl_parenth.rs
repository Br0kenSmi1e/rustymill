use std::collections::{HashMap, HashSet};

use num::rational::Ratio;

use crate::repr::{Index, IndexId, RangeId, TensorDef, Term};

pub type FactorSubset = u64;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LastStepIndices {
    pub left_ext: u64,
    pub right_ext: u64,
    pub sums: Vec<RangeId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TermSplit {
    pub left_sub_term: Term,
    pub right_sub_term: Term,
    pub last_step: LastStepIndices,
}

struct TermIndexInfo {
    factor_sum_bits: Vec<u64>,
    factor_ext_bits: Vec<u64>,
}

fn build_term_index_info(term: &Term, def: &TensorDef) -> TermIndexInfo {
    let mut sum_id_to_bit: HashMap<IndexId, usize> = HashMap::new();
    for (bit, idx) in term.sum_indices.iter().enumerate() {
        sum_id_to_bit.insert(idx.id, bit);
    }

    let mut ext_id_to_bit: HashMap<IndexId, usize> = HashMap::new();
    for (bit, idx) in def.ext_indices.iter().enumerate() {
        ext_id_to_bit.insert(idx.id, bit);
    }

    let mut factor_sum_bits = vec![0u64; term.factors.len()];
    let mut factor_ext_bits = vec![0u64; term.factors.len()];

    for (fi, factor) in term.factors.iter().enumerate() {
        for &idx_id in &factor.indices {
            if let Some(&bit) = sum_id_to_bit.get(&idx_id) {
                factor_sum_bits[fi] |= 1u64 << bit;
            }
            if let Some(&bit) = ext_id_to_bit.get(&idx_id) {
                factor_ext_bits[fi] |= 1u64 << bit;
            }
        }
    }

    TermIndexInfo {
        factor_sum_bits,
        factor_ext_bits,
    }
}

fn subset_sum_bits(info: &TermIndexInfo, subset: FactorSubset) -> u64 {
    let mut bits = 0u64;
    let mut s = subset;
    while s != 0 {
        let i = s.trailing_zeros() as usize;
        bits |= info.factor_sum_bits[i];
        s &= s - 1;
    }
    bits
}

fn subset_ext_bits(info: &TermIndexInfo, subset: FactorSubset) -> u64 {
    let mut bits = 0u64;
    let mut s = subset;
    while s != 0 {
        let i = s.trailing_zeros() as usize;
        bits |= info.factor_ext_bits[i];
        s &= s - 1;
    }
    bits
}

fn make_sub_term(term: &Term, subset: FactorSubset) -> Term {
    let mut factors = Vec::new();
    let mut s = subset;
    while s != 0 {
        let i = s.trailing_zeros() as usize;
        factors.push(term.factors[i].clone());
        s &= s - 1;
    }

    let mut present_ids: HashSet<IndexId> = HashSet::new();
    for factor in &factors {
        for &idx_id in &factor.indices {
            present_ids.insert(idx_id);
        }
    }

    let sum_indices: Vec<Index> = term
        .sum_indices
        .iter()
        .filter(|idx| present_ids.contains(&idx.id))
        .cloned()
        .collect();

    Term {
        coeff: Ratio::from_integer(1),
        sum_indices,
        factors,
    }
}

fn contracted_sum_ranges(
    term: &Term,
    left_sum_bits: u64,
    right_sum_bits: u64,
) -> Vec<RangeId> {
    let mut sums = Vec::new();
    let mut bits = left_sum_bits & right_sum_bits;
    while bits != 0 {
        let bit = bits.trailing_zeros() as usize;
        sums.push(term.sum_indices[bit].range);
        bits &= bits - 1;
    }
    sums.sort();
    sums
}

fn normalize_split(
    mut left_subset: FactorSubset,
    mut right_subset: FactorSubset,
    mut left_ext: u64,
    mut right_ext: u64,
) -> (FactorSubset, FactorSubset, u64, u64) {
    if left_ext > right_ext {
        std::mem::swap(&mut left_subset, &mut right_subset);
        std::mem::swap(&mut left_ext, &mut right_ext);
    }
    (left_subset, right_subset, left_ext, right_ext)
}

pub fn enumerate_splits(term: &Term, def: &TensorDef) -> Vec<TermSplit> {
    let n = term.factors.len();
    if n < 2 {
        return Vec::new();
    }

    let info = build_term_index_info(term, def);
    let full = (1u64 << n) - 1;
    let mut splits = Vec::new();

    let mut left = (full - 1) & full;
    while left != 0 {
        let right = full ^ left;
        if left < right {
            let left_ext = subset_ext_bits(&info, left);
            let right_ext = subset_ext_bits(&info, right);
            let (left, right, left_ext, right_ext) =
                normalize_split(left, right, left_ext, right_ext);

            let left_sum_bits = subset_sum_bits(&info, left);
            let right_sum_bits = subset_sum_bits(&info, right);
            let sums = contracted_sum_ranges(term, left_sum_bits, right_sum_bits);

            splits.push(TermSplit {
                left_sub_term: make_sub_term(term, left),
                right_sub_term: make_sub_term(term, right),
                last_step: LastStepIndices {
                    left_ext,
                    right_ext,
                    sums,
                },
            });
        }
        left = (left - 1) & full;
    }

    splits.sort_by(|a, b| {
        a.last_step
            .left_ext
            .cmp(&b.last_step.left_ext)
            .then(a.last_step.right_ext.cmp(&b.last_step.right_ext))
            .then(a.last_step.sums.cmp(&b.last_step.sums))
            .then(a.left_sub_term.factors.len().cmp(&b.left_sub_term.factors.len()))
            .then(a.right_sub_term.factors.len().cmp(&b.right_sub_term.factors.len()))
    });

    splits
}
