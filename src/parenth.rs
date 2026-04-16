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

/// All parenthesizations for a factor subset with a given set of extized sums.
#[derive(Clone, Debug)]
pub struct Interm {
    /// Internal contractable sums (all_sums & !extized).
    pub sum_indices: u64,
    /// Definition ext indices involved in this subset.
    pub ext_indices: u64,
    /// Sums forced external by parent (free sum indices of this intermediate).
    pub extized_sums: u64,
    pub evals: Vec<Eval>,
    pub best_cost: u64,
}

/// Full parenthesization result for one term.
#[derive(Clone, Debug)]
pub struct ParenthResult {
    /// Keyed by (factor_subset, extized_sums).
    pub memoir: HashMap<(FactorSubset, u64), Interm>,
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

/// Parenthesize a term: find all valid contraction trees via top-down
/// recursive DP with memoization, matching libparenth's semantics.
///
/// The DP is keyed on `(factor_subset, extized_sums)` where `extized_sums`
/// is the set of sum indices forced external by the parent split.
pub fn parenthesize(term: &Term, ext_indices: &[Index], ranges: &[Range]) -> ParenthResult {
    let info = IndexInfo::new(term, ext_indices, ranges);
    let n = info.n_factors;
    let mut memoir: HashMap<(FactorSubset, u64), Interm> = HashMap::new();

    if n == 0 {
        return ParenthResult { memoir, info };
    }

    let full_mask = (1u64 << n) - 1;
    solve(&info, full_mask, 0, &mut memoir);

    ParenthResult { memoir, info }
}

/// Top-down recursive solver. Returns the best cost for the given subset
/// with the given set of extized (forced-external) sums.
fn solve(
    info: &IndexInfo,
    subset: FactorSubset,
    extized: u64,
    memoir: &mut HashMap<(FactorSubset, u64), Interm>,
) -> u64 {
    if let Some(interm) = memoir.get(&(subset, extized)) {
        return interm.best_cost;
    }

    let all_sums = info.sum_bits(subset);
    let internal_sums = all_sums & !extized;
    let ext_bits = info.ext_bits(subset);

    if subset.count_ones() <= 1 {
        memoir.insert((subset, extized), Interm {
            sum_indices: internal_sums,
            ext_indices: ext_bits,
            extized_sums: all_sums & extized,
            evals: Vec::new(),
            best_cost: 0,
        });
        return 0;
    }

    // ext_size for this level: product of def ext dims + extized sum dims
    // (matching libparenth: exts = def_ext & involved | extized)
    let ext_size = (info.size_product_ext(ext_bits)
        * info.size_product_sum(all_sums & extized))
        .max(1);

    let mut evals = Vec::new();
    let mut best_cost = u64::MAX;

    // Enumerate all binary splits: subset = left | right
    // Only consider left < right to avoid duplicates.
    let mut sub = (subset - 1) & subset;
    while sub != 0 {
        let left = sub;
        let right = subset ^ sub;
        if left < right {
            let sums_on_left = info.sum_bits(left);
            let sums_on_right = info.sum_bits(right);
            // Contracted sums: shared between both parts AND internal (not extized).
            let contracted = sums_on_left & sums_on_right & internal_sums;

            let child_extized = extized | contracted;
            let left_cost = solve(info, left, child_extized, memoir);
            let right_cost = solve(info, right, child_extized, memoir);

            // Step cost matching libparenth: lsc only, no + ext_size.
            let sum_size = info.size_product_sum(contracted).max(1);
            let lsc = if sum_size == 1 {
                ext_size
            } else {
                2 * ext_size * sum_size
            };
            let total = lsc + left_cost + right_cost;

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

    memoir.insert((subset, extized), Interm {
        sum_indices: internal_sums,
        ext_indices: ext_bits,
        extized_sums: all_sums & extized,
        evals,
        best_cost,
    });

    best_cost
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
    let mut subset_to_tensor: HashMap<(FactorSubset, u64), TensorId> = HashMap::new();

    // Map single factors to their original tensor IDs (extized doesn't matter for leaves)
    for (i, factor) in term.factors.iter().enumerate() {
        // Leaves may be looked up with any extized value, so we insert a sentinel.
        // We'll handle leaf lookup specially below.
        let _ = factor; // suppress unused warning; we use factor.tensor below
        let _ = i;
    }

    // Build index mappings: bit position -> (IndexId, RangeId)
    let sum_index_map: Vec<(IndexId, RangeId)> = term.sum_indices.iter()
        .map(|idx| (idx.id, idx.range))
        .collect();
    let ext_index_map: Vec<(IndexId, RangeId)> = ext_indices.iter()
        .map(|idx| (idx.id, idx.range))
        .collect();

    // Collect (subset, extized) pairs in the optimal eval tree, process small to large
    let mut pairs_to_process: Vec<(FactorSubset, u64)> = Vec::new();
    collect_optimal_subsets(result, full_mask, 0, &mut pairs_to_process);
    pairs_to_process.retain(|(s, _)| s.count_ones() >= 2);
    pairs_to_process.sort_by_key(|(s, _)| s.count_ones());
    pairs_to_process.dedup();

    for &(subset, extized) in &pairs_to_process {
        let interm = &result.memoir[&(subset, extized)];
        let best_eval = interm.evals.iter().min_by_key(|e| e.cost).unwrap();

        let child_extized = extized | best_eval.contracted_sums;

        let left_tensor = get_tensor_id(
            &subset_to_tensor, &term.factors, best_eval.left, child_extized,
        );
        let right_tensor = get_tensor_id(
            &subset_to_tensor, &term.factors, best_eval.right, child_extized,
        );

        // Free indices of this intermediate: def ext + extized sums
        let alive_exts = interm.ext_indices;
        let alive_extized = interm.extized_sums;

        let mut def_ext_indices = Vec::new();
        let mut m = alive_exts;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            def_ext_indices.push(Index {
                id: ext_index_map[bit].0,
                range: ext_index_map[bit].1,
            });
            m &= m - 1;
        }
        let mut m = alive_extized;
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
        let mut m = best_eval.contracted_sums;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            step_sums.push(Index {
                id: sum_index_map[bit].0,
                range: sum_index_map[bit].1,
            });
            m &= m - 1;
        }

        // Collect indices for left and right operand references
        let left_interm = &result.memoir[&(best_eval.left, child_extized)];
        let right_interm = &result.memoir[&(best_eval.right, child_extized)];
        let left_indices = collect_operand_indices(left_interm, &sum_index_map, &ext_index_map);
        let right_indices = collect_operand_indices(right_interm, &sum_index_map, &ext_index_map);

        // Create new intermediate tensor
        let new_tensor = comp.add_tensor(vec![]);

        let full_mask = (1u64 << result.info.n_factors) - 1;
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

        subset_to_tensor.insert((subset, extized), new_tensor);
        defs.push(def);
    }

    defs
}

/// Get the TensorId for a subset. For single-factor leaves, return the original tensor.
fn get_tensor_id(
    subset_to_tensor: &HashMap<(FactorSubset, u64), TensorId>,
    factors: &[Factor],
    subset: FactorSubset,
    extized: u64,
) -> TensorId {
    if subset.count_ones() == 1 {
        let idx = subset.trailing_zeros() as usize;
        factors[idx].tensor
    } else {
        subset_to_tensor[&(subset, extized)]
    }
}

/// Recursively collect all (subset, extized) pairs in the optimal eval tree.
fn collect_optimal_subsets(
    result: &ParenthResult,
    subset: FactorSubset,
    extized: u64,
    out: &mut Vec<(FactorSubset, u64)>,
) {
    if subset.count_ones() <= 1 {
        return;
    }
    out.push((subset, extized));
    let interm = &result.memoir[&(subset, extized)];
    let best_eval = interm.evals.iter().min_by_key(|e| e.cost).unwrap();
    let child_extized = extized | best_eval.contracted_sums;
    collect_optimal_subsets(result, best_eval.left, child_extized, out);
    collect_optimal_subsets(result, best_eval.right, child_extized, out);
}

/// Collect the IndexIds for an operand (its ext + extized sum indices).
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

    let mut m = interm.extized_sums;
    while m != 0 {
        let bit = m.trailing_zeros() as usize;
        indices.push(sum_map[bit].0);
        m &= m - 1;
    }

    indices
}
