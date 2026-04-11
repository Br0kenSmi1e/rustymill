use std::collections::HashMap;

use crate::repr::{Index, IndexId, Range, Term};

/// Bitmask of factors in a subset (up to 64 factors).
pub type FactorSubset = u64;

/// One way to split a factor subset into two operands.
#[derive(Clone, Debug)]
pub struct Eval {
    pub left: FactorSubset,
    pub right: FactorSubset,
    pub contracted_sums: u64,
    pub cost: u64,
}

/// All parenthesizations for a factor subset.
#[derive(Clone, Debug)]
pub struct Interm {
    pub sum_indices: u64,
    pub ext_indices: u64,
    pub evals: Vec<Eval>,
    pub best_cost: u64,
}

/// Full parenthesization result for one term.
#[derive(Clone, Debug)]
pub struct ParenthResult {
    pub memoir: HashMap<FactorSubset, Interm>,
    pub info: IndexInfo,
}

/// Precomputed index information for a term.
#[derive(Clone, Debug)]
pub struct IndexInfo {
    pub n_factors: usize,
    pub factor_sum_indices: Vec<u64>,
    pub factor_ext_indices: Vec<u64>,
    pub sum_sizes: Vec<u64>,
    pub ext_sizes: Vec<u64>,
}

impl IndexInfo {
    pub fn new(term: &Term, ext_indices: &[Index], ranges: &[Range]) -> Self {
        let mut sum_id_to_bit: HashMap<IndexId, usize> = HashMap::new();
        let mut sum_sizes: Vec<u64> = Vec::new();
        for idx in &term.sum_indices {
            let bit = sum_id_to_bit.len();
            sum_id_to_bit.insert(idx.id, bit);
            sum_sizes.push(ranges[idx.range.0 as usize].size);
        }

        let mut ext_id_to_bit: HashMap<IndexId, usize> = HashMap::new();
        let mut ext_sizes: Vec<u64> = Vec::new();
        for idx in ext_indices {
            let bit = ext_id_to_bit.len();
            ext_id_to_bit.insert(idx.id, bit);
            ext_sizes.push(ranges[idx.range.0 as usize].size);
        }

        let n_factors = term.factors.len();
        let mut factor_sum_indices = vec![0u64; n_factors];
        let mut factor_ext_indices = vec![0u64; n_factors];

        for (fi, factor) in term.factors.iter().enumerate() {
            for &idx_id in &factor.indices {
                if let Some(&bit) = sum_id_to_bit.get(&idx_id) {
                    factor_sum_indices[fi] |= 1u64 << bit;
                }
                if let Some(&bit) = ext_id_to_bit.get(&idx_id) {
                    factor_ext_indices[fi] |= 1u64 << bit;
                }
            }
        }

        IndexInfo {
            n_factors,
            factor_sum_indices,
            factor_ext_indices,
            sum_sizes,
            ext_sizes,
        }
    }

    pub fn sum_bits(&self, subset: FactorSubset) -> u64 {
        let mut bits = 0u64;
        let mut s = subset;
        while s != 0 {
            let i = s.trailing_zeros() as usize;
            bits |= self.factor_sum_indices[i];
            s &= s - 1;
        }
        bits
    }

    pub fn ext_bits(&self, subset: FactorSubset) -> u64 {
        let mut bits = 0u64;
        let mut s = subset;
        while s != 0 {
            let i = s.trailing_zeros() as usize;
            bits |= self.factor_ext_indices[i];
            s &= s - 1;
        }
        bits
    }

    pub fn size_product_sum(&self, mask: u64) -> u64 {
        let mut product = 1u64;
        let mut m = mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            product *= self.sum_sizes[i];
            m &= m - 1;
        }
        product
    }

    pub fn size_product_ext(&self, mask: u64) -> u64 {
        let mut product = 1u64;
        let mut m = mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            product *= self.ext_sizes[i];
            m &= m - 1;
        }
        product
    }
}

/// Result of computing the cost of a single binary split.
#[derive(Clone, Debug)]
pub struct SplitCost {
    pub contracted_sums: u64,
    pub step_cost: u64,
}

/// Compute the step cost of splitting a factor subset into left and right.
pub fn split_cost(info: &IndexInfo, left: FactorSubset, right: FactorSubset) -> SplitCost {
    let all = left | right;
    let sum_left = info.sum_bits(left);
    let sum_right = info.sum_bits(right);
    let contracted = sum_left & sum_right;
    let uncontracted_sums = info.sum_bits(all) & !contracted;

    let ext_of_result = info.ext_bits(all);
    let ext_size = info.size_product_ext(ext_of_result)
        * info.size_product_sum(uncontracted_sums);
    let ext_size = ext_size.max(1);

    let sum_size = info.size_product_sum(contracted);
    let sum_size = sum_size.max(1);

    let step_cost = if sum_size == 1 {
        ext_size
    } else {
        2 * ext_size * sum_size
    } + ext_size;

    SplitCost {
        contracted_sums: contracted,
        step_cost,
    }
}
