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

/// Parenthesize a term: find all valid contraction trees via exhaustive subset DP.
pub fn parenthesize(term: &Term, ext_indices: &[Index], ranges: &[Range]) -> ParenthResult {
    let info = IndexInfo::new(term, ext_indices, ranges);
    let n = info.n_factors;
    let mut memoir: HashMap<FactorSubset, Interm> = HashMap::new();

    // Base cases: single factors
    for i in 0..n {
        let mask = 1u64 << i;
        memoir.insert(mask, Interm {
            sum_indices: info.factor_sum_indices[i],
            ext_indices: info.factor_ext_indices[i],
            evals: Vec::new(),
            best_cost: 0,
        });
    }

    // Process subsets in order of increasing popcount (size 2, 3, ..., n)
    let full_mask = (1u64 << n) - 1;
    for size in 2..=n {
        for subset in SubsetIter::of_size(full_mask, size) {
            // Compute the still-open sum indices for this subset using XOR-based counting:
            // An index is open iff it appears in an odd number of factors in the subset.
            let sum_idx = {
                let mut open = 0u64;
                let mut s = subset;
                while s != 0 {
                    let i = s.trailing_zeros() as usize;
                    open ^= info.factor_sum_indices[i];
                    s &= s - 1;
                }
                open
            };
            let ext_idx = info.ext_bits(subset);
            let mut evals = Vec::new();
            let mut best_cost = u64::MAX;

            // Enumerate all binary splits: subset = left | right
            // Only consider left < right to avoid duplicates
            let mut sub = (subset - 1) & subset;
            while sub != 0 {
                let left = sub;
                let right = subset ^ sub;
                if left < right {
                    let left_sum = memoir[&left].sum_indices;
                    let right_sum = memoir[&right].sum_indices;
                    let contracted = left_sum & right_sum;
                    let uncontracted_sums = (left_sum | right_sum) & !contracted;

                    let ext_of_result = info.ext_bits(subset);
                    let ext_size = (info.size_product_ext(ext_of_result)
                        * info.size_product_sum(uncontracted_sums))
                        .max(1);
                    let sum_size = info.size_product_sum(contracted).max(1);
                    let step_cost = if sum_size == 1 {
                        ext_size
                    } else {
                        2 * ext_size * sum_size
                    } + ext_size;

                    let left_best = memoir[&left].best_cost;
                    let right_best = memoir[&right].best_cost;
                    let total = step_cost + left_best + right_best;

                    evals.push(Eval {
                        left,
                        right,
                        contracted_sums: contracted,
                        cost: total,
                    });

                    if total < best_cost {
                        best_cost = total;
                    }
                }
                sub = (sub - 1) & subset;
            }

            memoir.insert(subset, Interm {
                sum_indices: sum_idx,
                ext_indices: ext_idx,
                evals,
                best_cost,
            });
        }
    }

    ParenthResult { memoir, info }
}

/// Iterator over all subsets of a given mask with a specific popcount.
struct SubsetIter {
    mask: u64,
    target_size: u32,
    current: u64,
    done: bool,
}

impl SubsetIter {
    fn of_size(mask: u64, size: usize) -> Self {
        let target_size = size as u32;
        let mut current = 0u64;
        let mut count = 0;
        let mut m = mask;
        while m != 0 && count < target_size {
            let lowest = m & m.wrapping_neg();
            current |= lowest;
            m ^= lowest;
            count += 1;
        }
        let done = count < target_size;
        SubsetIter { mask, target_size, current, done }
    }
}

impl Iterator for SubsetIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.done {
            return None;
        }
        let result = self.current;

        let c = self.current;
        let m = self.mask;

        let mut positions = Vec::new();
        let mut tmp = m;
        let mut idx = 0;
        while tmp != 0 {
            let lowest = tmp & tmp.wrapping_neg();
            if c & lowest != 0 {
                positions.push(idx);
            }
            tmp ^= lowest;
            idx += 1;
        }

        let k = positions.len();
        let n = m.count_ones() as usize;

        let mut i = k;
        loop {
            if i == 0 {
                self.done = true;
                return Some(result);
            }
            i -= 1;
            if positions[i] < n - (k - i) {
                break;
            }
        }

        positions[i] += 1;
        for j in (i + 1)..k {
            positions[j] = positions[j - 1] + 1;
        }

        let mask_bits: Vec<u64> = {
            let mut bits = Vec::new();
            let mut tmp = m;
            while tmp != 0 {
                let lowest = tmp & tmp.wrapping_neg();
                bits.push(lowest);
                tmp ^= lowest;
            }
            bits
        };

        let mut next = 0u64;
        for &pos in &positions {
            next |= mask_bits[pos];
        }
        self.current = next;

        Some(result)
    }
}
