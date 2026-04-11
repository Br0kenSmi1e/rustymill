# Sum Optimization (Biclique Factorization) Design

## Context

This is the second component of rustymill's optimization pipeline. Given a `TensorDef` whose terms have been parenthesized, find all profitable single-step factorizations by detecting bicliques in constriction graphs. Each factorization applies one biclique — the caller decides whether to apply it and iterate.

Depends on: `repr.rs` (types), `parenth.rs` (ParenthResult), `canon.rs` (canonicalization), `cost.rs` (cost model).

## Design Decisions

- **Single-step factorization.** `factorizations()` finds all profitable bicliques but does not recurse. The caller (greedy loop or MCTS) controls iteration.
- **Decoupled from parenthesization.** Caller passes pre-parenthesized results. Swapping parenthesization strategies doesn't affect this module.
- **Faithful Bron-Kerbosch.** Reproduces gristmill's biclique enumeration with coefficient compatibility checking.
- **Rational coefficient arithmetic.** Exact comparison, no floating-point tolerance.

## Public Interface

```rust
/// Find all profitable single-step factorizations of a TensorDef.
pub fn factorizations(
    def: &TensorDef,
    parenth_results: &[ParenthResult],  // one per term
    ranges: &[Range],
    tensors: &[TensorInfo],
) -> Vec<Factorization>;

/// One profitable factorization (one biclique applied).
pub struct Factorization {
    /// Which terms in the original TensorDef are consumed.
    pub terms_consumed: Vec<usize>,
    /// New intermediate TensorDefs to add (0-2: left sum, right sum).
    pub intermediates: Vec<TensorDef>,
    /// The term that replaces the consumed terms.
    pub replacement_term: Term,
    /// Cost saving (positive = beneficial).
    pub saving: i64,
}
```

## Algorithm Overview

1. Build constriction graphs from parenthesized terms
2. For each graph, run Bron-Kerbosch to enumerate all maximal profitable bicliques
3. Convert each biclique to a `Factorization`

## Constriction Graph Construction

For each term `i` in `def.terms`, and each eval `j` in `parenth_results[i].memoir[full_set].evals`:

1. The eval gives a binary split `(left_subset, right_subset)` of factors.
2. Determine the canonical form of each side's factor content (using `canon.rs`).
3. Determine which external indices each side involves (bitmask from `parenth_results[i].info`).
4. Group by index pattern into separate constriction graphs.
5. Add an edge `(left_vertex, right_vertex)` with metadata.

### Data Structures

```rust
/// Index pattern key for a constriction graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LastStepIndices {
    left_ext: u64,     // bitmask of ext indices on left side
    right_ext: u64,    // bitmask of ext indices on right side
    sums: u64,         // bitmask of summation indices contracted at this step
}

type VertexId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side { Left, Right }

/// Metadata for an edge in the constriction graph.
struct EdgeInfo {
    term_idx: usize,       // which term this edge comes from
    eval_idx: usize,       // which eval of that term's full-set parenthesization
    coeff: Rational,       // coefficient on this edge
    exc_cost: i64,         // eval_cost - best_cost for this term (excess cost)
}

/// A bipartite constriction graph for one index pattern.
struct ConstrGraph {
    /// Canonical form → vertex id mapping.
    canon_to_vertex: HashMap<CanonTerm, VertexId>,
    /// Which side each vertex is on.
    vertex_side: Vec<Side>,
    /// Adjacency: for each vertex, map of neighbor → list of edges.
    adj: Vec<HashMap<VertexId, Vec<EdgeInfo>>>,
    /// The index pattern this graph corresponds to.
    last_step: LastStepIndices,
}
```

### Vertex Identification

Two factors from different terms become the **same vertex** if they have the same canonical form (after canonicalization via `canon_term()`). This is how shared structure across terms is detected.

### Multiple Constriction Graphs

Different evals of the same term may produce different index patterns (which ext indices are on left vs right). Each unique `LastStepIndices` gets its own graph. Bicliques can only form within a single graph (same index pattern).

## Bron-Kerbosch Biclique Enumeration

### Coefficient Model

In a biclique `{L1, L2, ...} × {R1, R2, ...}`, each edge represents a term:
```
edge(Li, Rj) → coeff_ij * Li * Rj
```

The biclique requires all coefficients to factor as:
```
coeff_ij = leading_coeff * left_coeff_i * right_coeff_j
```

This is checked incrementally:

1. **First cross-part edge**: edge coefficient becomes `leading_coeff`. Both vertices get `coeff = 1`.
2. **Adding same-part vertex** (e.g., L2 when L1, R1 exist): `L2.coeff = edge_coeff(L2,R1) / leading_coeff`.
3. **Subsequent cross-part edges**: check `edge_coeff == leading_coeff * Li.coeff * Rj.coeff`. Reject if not.

With rational arithmetic, comparison is exact.

### State

```rust
struct BronKerbosch<'a> {
    graph: &'a ConstrGraph,
    // Current biclique being built
    left_verts: Vec<(VertexId, Rational)>,
    right_verts: Vec<(VertexId, Rational)>,
    leading_coeff: Option<Rational>,
    terms_used: u64,           // bitmask of terms in biclique
    savings_stack: Vec<i64>,   // cumulative savings at each depth
    // Cost coefficients for this graph's index pattern
    cost_coeffs: CostCoeffs,
}

struct Delta {
    coeff: Rational,
    leading_coeff: Option<Rational>,
    terms: u64,           // bitmask of terms contributed
    exc_cost: i64,
    saving: i64,
}

struct CostCoeffs {
    final_cost: u64,      // cost of final contraction + addition
    prep_left: u64,       // cost of summing left factors
    prep_right: u64,      // cost of summing right factors
}
```

### Core Algorithm: `expand()`

Recursive backtracking that yields maximal profitable bicliques:

```
expand(subgraph, candidates):
    check if current biclique is maximal (no candidate can be added with positive saving)
    if maximal AND profitable (both sides non-empty, at least one side > 1, saving >= 0):
        yield this biclique

    for each candidate vertex q:
        compute delta for adding q (update_delta for each existing vertex)
        if compatible:
            add q to biclique
            recurse with filtered subgraph and candidates
            remove q (backtrack)
```

### `update_delta()` — The Heart

For a candidate vertex `new_v` and an existing vertex `curr_v`:

**Same part** (both left or both right):
- If `new_v` has `leading_coeff`: compute `coeff = curr_leading_coeff / new_leading_coeff`
- Otherwise: no constraint (same-part vertices don't need edges between them)

**Different parts**:
- Must have an edge between them (otherwise can't form biclique)
- Check term disjointness (no term used twice)
- Accumulate `exc_cost` from edge
- Check coefficient compatibility:
  - If `new_v` has `leading_coeff`: `coeff = edge_coeff / new_leading_coeff`
  - If no global `leading_coeff` yet: set it from this edge
  - Otherwise: verify `edge_coeff == leading_coeff * new_v.coeff * curr_v.coeff`

Return `None` if any check fails (vertex incompatible).

### Saving Formula

```
final_cost = contraction_cost(ext_size, sum_size) + ext_size
prep_left = product of sizes of (left_ext_indices ∪ sum_indices)
prep_right = product of sizes of (right_ext_indices ∪ sum_indices)

// Marginal gross saving when adding a vertex:
gross_for_new_left_vertex = n_right * final_cost - prep_left
gross_for_new_right_vertex = n_left * final_cost - prep_right

// Actual saving:
saving = gross - accumulated_exc_cost
```

Where `contraction_cost(ext, sum)` follows our existing cost model:
- `if sum == 1: ext else: 2 * ext * sum`

## Factorization Conversion

Converting a biclique to a `Factorization`:

1. **Terms consumed**: union of all term indices from edges in the biclique.

2. **Left intermediate** (if n_left > 1): a new `TensorDef` whose terms are the left vertices scaled by their coefficients. External indices = left_ext ∪ sums.

3. **Right intermediate** (if n_right > 1): same for right vertices. External indices = right_ext ∪ sums.

4. **Replacement term**: `leading_coeff * left_factor * right_factor`, where left/right factors reference the intermediates (or original tensors if that side has only 1 vertex). Summation indices = the contracted sums from `LastStepIndices`.

5. **Saving**: from the biclique's saving computation.

## Module Structure

One new file: `src/constr.rs`

- `LastStepIndices`, `ConstrGraph`, `EdgeInfo`, `Side`, `VertexId` — graph types
- `build_constr_graphs()` — construct graphs from parenthesized terms
- `BronKerbosch` — biclique enumeration with coefficient checking
- `CostCoeffs` — saving computation helpers
- `Factorization` — result type
- `factorizations()` — public entry point

## What This Design Does NOT Cover

- Greedy loop / MCTS orchestration (caller's responsibility)
- Recursion after applying a factorization (caller applies and re-parenthesizes)
- Symmetry optimization (`_optimize_common_symmtrization` in gristmill)
- Linearization (converting evaluation DAG back to TensorDef sequence)
