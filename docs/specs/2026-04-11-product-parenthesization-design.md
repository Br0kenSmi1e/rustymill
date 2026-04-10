# Product Parenthesization Design

## Context

This is the first component of rustymill's optimization pipeline. Given a tensor product term with N factors, find all valid binary contraction trees (parenthesizations) with their FLOP costs. Results are stored so both product optimization (pick the optimal) and future sum optimization (use all alternatives) can consume them.

## Design Decisions

- **Exhaustive enumeration.** Try all binary splits for every factor subset. No pruning. Simple subset DP with bitmasks.
- **Store all parenthesizations.** Not just the optimal — sum optimization needs alternatives to build constriction graph edges with `exc_cost = eval_cost - optimal_cost`.
- **No dimension chunking.** For 2-6 factors, the naive approach is fast enough. Chunking can be added later as an optimization.
- **Works on our `TensorComputation` representation.** Input is a `Term` + `ext_indices` + `ranges`.

## Algorithm

Subset DP over factor bitmasks. For N factors, each subset S is a `u64` bitmask.

**Base case:** |S| = 1 → cost = 0, no evaluations.

**Recursive case:** For each subset S with |S| >= 2:

1. Compute which summation and external indices are "alive" in S (union of per-factor index sets).
2. Enumerate all binary splits S = L ∪ R where L, R are non-empty and L < R (avoid symmetric duplicates).
3. For each split:
   - Contracted summation indices = those appearing in both L and R.
   - External indices of result = alive indices minus contracted sums.
   - Step cost: if no contracted sums, `ext_size`; otherwise `2 * ext_size * sum_size + ext_size`.
   - Total cost = step_cost + best_cost(L) + best_cost(R).
   - Store `Eval { left: L, right: R, sums: contracted, cost: total }`.
4. Set `best_cost(S) = min(eval.cost for eval in evals)`.

Process subsets in order of increasing popcount so dependencies are resolved.

## Data Structures

```rust
/// Bitmask of factors in a subset.
pub type FactorSubset = u64;

/// One way to split a factor subset into two operands.
pub struct Eval {
    pub left: FactorSubset,
    pub right: FactorSubset,
    pub contracted_sums: Vec<usize>,  // indices into sum_sizes
    pub cost: u64,
}

/// All parenthesizations for a factor subset.
pub struct Interm {
    pub sum_indices: u64,    // bitmask: which sum indices are alive
    pub ext_indices: u64,    // bitmask: which ext indices are alive
    pub evals: Vec<Eval>,
    pub best_cost: u64,      // min cost among evals (0 for single factor)
}

/// Full parenthesization result for one term.
pub struct ParenthResult {
    pub memoir: HashMap<FactorSubset, Interm>,
    pub n_factors: usize,
    /// Per-factor bitmask of which summation indices it uses.
    pub factor_sum_indices: Vec<u64>,
    /// Per-factor bitmask of which external indices it uses.
    pub factor_ext_indices: Vec<u64>,
    /// Sizes of summation indices (index into these via bit position).
    pub sum_sizes: Vec<u64>,
    /// Sizes of external indices.
    pub ext_sizes: Vec<u64>,
}
```

## Index Precomputation

Before DP, analyze the term to build per-factor index bitmasks:

1. Collect all unique summation indices from `term.sum_indices`. Assign each a bit position. Record sizes.
2. Collect all unique external indices from `ext_indices`. Assign each a bit position. Record sizes.
3. For each factor, scan its `indices` to determine which summation and external bits it uses.

For a subset S:
- `sum_bits(S) = OR of factor_sum_indices[i] for i in S`
- `ext_bits(S) = OR of factor_ext_indices[i] for i in S`
- Contracted sums for split (L, R) = `sum_bits(L) & sum_bits(R)`

## Cost Computation

For a binary split (L, R) with contracted summation set C:

```
ext_of_result = ext_bits(L | R) | (sum_bits(L | R) & !C)
ext_size = product of sizes for bits in ext_of_result (min 1)
sum_size = product of sizes for bits in C (min 1)

step_cost = if sum_size == 1 { ext_size } else { 2 * ext_size * sum_size } + ext_size
total_cost = step_cost + best_cost(L) + best_cost(R)
```

Note: `ext_of_result` includes both the original external indices AND any summation indices not yet contracted (they become external from the perspective of the intermediate).

## Public Interface

```rust
/// Parenthesize a term: find all valid contraction trees.
pub fn parenthesize(
    term: &Term,
    ext_indices: &[Index],
    ranges: &[Range],
) -> ParenthResult;

/// Extract the optimal contraction as new TensorDefs added to a computation.
/// Returns the TensorId of the final result intermediate.
pub fn extract_optimal(
    result: &ParenthResult,
    term: &Term,
    ext_indices: &[Index],
    comp: &mut TensorComputation,
) -> TensorId;
```

## Module Structure

One new file: `src/parenth.rs`

## What This Design Does NOT Cover

- Dimension chunking optimization (can be added later for performance)
- Pruning modes (GREEDY, OPT — only affects which evals are kept)
- Sum optimization (future consumer of `ParenthResult.memoir`)
- Integration into the full optimization pipeline
