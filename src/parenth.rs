use std::collections::HashMap;

use crate::repr::{Factor, Index, IndexId, Range, RangeId, TensorComputation, TensorDef, TensorId, Term};
use num::rational::Ratio;

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

/// Extract the optimal contraction tree as a sequence of TensorDefs.
///
/// Walks the optimal eval chain from the full factor set down to individual factors,
/// creating intermediate TensorDefs for each binary contraction step.
/// Returns the new definitions in evaluation order (dependencies first).
pub fn extract_optimal(
    result: &ParenthResult,
    term: &Term,
    ext_indices: &[Index],
    comp: &mut TensorComputation,
) -> Vec<TensorDef> {
    let n = result.info.n_factors;
    if n <= 1 {
        return Vec::new();
    }

    let full_mask = (1u64 << n) - 1;
    let mut defs = Vec::new();
    let mut subset_to_tensor: HashMap<FactorSubset, TensorId> = HashMap::new();

    // Map single factors to their original tensor IDs
    for (i, factor) in term.factors.iter().enumerate() {
        subset_to_tensor.insert(1u64 << i, factor.tensor);
    }

    // Build index mappings: bit position -> (IndexId, RangeId)
    let sum_index_map: Vec<(IndexId, RangeId)> = term.sum_indices.iter()
        .map(|idx| (idx.id, idx.range))
        .collect();
    let ext_index_map: Vec<(IndexId, RangeId)> = ext_indices.iter()
        .map(|idx| (idx.id, idx.range))
        .collect();

    // Collect subsets in the optimal eval tree, process small to large
    let mut subsets_to_process: Vec<FactorSubset> = Vec::new();
    collect_optimal_subsets(result, full_mask, &mut subsets_to_process);
    subsets_to_process.retain(|s| s.count_ones() >= 2);
    subsets_to_process.sort_by_key(|s| s.count_ones());
    subsets_to_process.dedup();

    for &subset in &subsets_to_process {
        let interm = &result.memoir[&subset];
        let best_eval = interm.evals.iter().min_by_key(|e| e.cost).unwrap();

        let left_tensor = subset_to_tensor[&best_eval.left];
        let right_tensor = subset_to_tensor[&best_eval.right];

        // External indices for this intermediate:
        // = original ext indices alive in this subset + uncontracted sum indices
        let contracted = best_eval.contracted_sums;
        let alive_exts = interm.ext_indices;
        let alive_sums = interm.sum_indices; // open (uncontracted) sums for this subset

        let mut def_ext_indices = Vec::new();
        // Add original external indices
        let mut m = alive_exts;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            def_ext_indices.push(Index {
                id: ext_index_map[bit].0,
                range: ext_index_map[bit].1,
            });
            m &= m - 1;
        }
        // Add uncontracted summation indices (they're "external" to this intermediate)
        let mut m = alive_sums;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            def_ext_indices.push(Index {
                id: sum_index_map[bit].0,
                range: sum_index_map[bit].1,
            });
            m &= m - 1;
        }

        // Contracted summation indices for this step
        let mut step_sums = Vec::new();
        let mut m = contracted;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            step_sums.push(Index {
                id: sum_index_map[bit].0,
                range: sum_index_map[bit].1,
            });
            m &= m - 1;
        }

        // Collect indices for left and right operand references
        let left_indices = collect_operand_indices(
            &result.memoir[&best_eval.left], &sum_index_map, &ext_index_map,
        );
        let right_indices = collect_operand_indices(
            &result.memoir[&best_eval.right], &sum_index_map, &ext_index_map,
        );

        // Create new intermediate tensor
        let slots: Vec<RangeId> = def_ext_indices.iter().map(|idx| idx.range).collect();
        let new_tensor = comp.add_tensor(&slots, vec![]);

        let new_term = Term {
            coeff: if subset == full_mask { term.coeff.clone() } else { Ratio::from_integer(1) },
            sum_indices: step_sums,
            factors: vec![
                Factor { tensor: left_tensor, indices: left_indices },
                Factor { tensor: right_tensor, indices: right_indices },
            ],
        };

        let def = TensorDef {
            base: new_tensor,
            ext_indices: def_ext_indices,
            terms: vec![new_term],
        };

        subset_to_tensor.insert(subset, new_tensor);
        defs.push(def);
    }

    defs
}

/// Recursively collect all subsets in the optimal eval tree.
fn collect_optimal_subsets(
    result: &ParenthResult,
    subset: FactorSubset,
    out: &mut Vec<FactorSubset>,
) {
    if subset.count_ones() <= 1 {
        return;
    }
    out.push(subset);
    let interm = &result.memoir[&subset];
    let best_eval = interm.evals.iter().min_by_key(|e| e.cost).unwrap();
    collect_optimal_subsets(result, best_eval.left, out);
    collect_optimal_subsets(result, best_eval.right, out);
}

/// Collect the IndexIds for an operand (its external + open sum indices).
fn collect_operand_indices(
    interm: &Interm,
    sum_map: &[(IndexId, RangeId)],
    ext_map: &[(IndexId, RangeId)],
) -> Vec<IndexId> {
    let mut indices = Vec::new();

    let mut m = interm.ext_indices;
    while m != 0 {
        let bit = m.trailing_zeros() as usize;
        indices.push(ext_map[bit].0);
        m &= m - 1;
    }

    let mut m = interm.sum_indices;
    while m != 0 {
        let bit = m.trailing_zeros() as usize;
        indices.push(sum_map[bit].0);
        m &= m - 1;
    }

    indices
}
