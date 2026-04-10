# Tensor Computation Representation Design

## Context

`rustymill` is a Rust rewrite of gristmill's tensor contraction optimizer. The primary motivation is long-term maintainability: replacing accumulated mutable state with a clean functional architecture using copy-on-write (`Rc`/`Arc`) for cheap state branching.

The representation is algorithm-agnostic. The same `TensorComputation` type serves as both input and output for any optimization algorithm (greedy, MCTS, future strategies). Algorithms are consumers of this representation, designed separately.

## Design Decisions

- **Pure Rust, no Python dependency.** No SymPy, no drudge at runtime.
- **Numeric representation.** Integer IDs for indices/tensors, integer dimension sizes, rational coefficients. No symbolic algebra.
- **Canonicalization in Rust.** Reimplements drudge's `term.canon()` over integer-ID structures.
- **Copy-on-write state management.** `Rc<T>` with `Rc::make_mut` for cheap branching in search algorithms. `Arc` for future multithreading.
- **JSON serde.** Rust builder API as primary interface, with JSON serialization/deserialization for interop.
- **Parenthesization rewritten in Rust.** No C++ FFI dependency on libparenth.

## Core Types

### Primitives

```rust
// Opaque IDs (newtypes over u32)
struct RangeId(u32);
struct IndexId(u32);
struct TensorId(u32);
```

### Range

A named index space with a known dimension.

```rust
struct Range {
    id: RangeId,
    size: u64,
}
```

### Index

An index variable scoped to a specific range.

```rust
struct Index {
    id: IndexId,
    range: RangeId,
}
```

### Tensor Symmetry

Symmetry is represented as generators of a permutation group acting on index slots. Each generator is a permutation plus an action (sign/conjugation change).

```rust
enum SymAction {
    Identity,
    Negate,
    Conjugate,
    NegateConjugate,
}

struct SymGenerator {
    perm: Vec<usize>,
    action: SymAction,
}
```

### TensorInfo

A named tensor with index slot structure and symmetry group.

```rust
struct TensorInfo {
    id: TensorId,
    slots: Vec<RangeId>,           // each slot expects indices from this range
    symmetry: Vec<SymGenerator>,   // generators of the symmetry group
}
```

### Factor

A tensor applied to concrete indices (one factor in a product).

```rust
struct Factor {
    tensor: TensorId,
    indices: Vec<IndexId>,
}
```

### Term

A rational coefficient times a product of factors, with its own summation indices.

```rust
struct Term {
    coeff: Rational,            // num::rational::Ratio<i64> or arbitrary precision
    sum_indices: Vec<Index>,    // summation indices for this term
    factors: Vec<Factor>,
}
```

### TensorDef

A tensor definition: `LHS[ext_indices] = term1 + term2 + ...`. External indices are shared across all terms (they must match the LHS). Each term brings its own summation indices.

```rust
struct TensorDef {
    base: TensorId,
    ext_indices: Vec<Index>,
    terms: Vec<Term>,
}
```

### TensorComputation

The top-level container. Both input and output of any optimization algorithm.

```rust
struct TensorComputation {
    ranges: Vec<Range>,
    tensors: Vec<TensorInfo>,
    definitions: Vec<TensorDef>,
}
```

Input: a few tensors and definitions. Output: same type with additional intermediate tensors and definitions, lower total cost.

## Pure Functions on the Representation

### Cost Model

```rust
/// Cost of evaluating one TensorDef (FLOP count).
///
/// For each term: contraction cost + addition into output.
/// - No summation: ext_size (copy/scale) + ext_size (addition)
/// - With summation: 2 * ext_size * sum_size (multiply-add) + ext_size (addition)
fn def_cost(def: &TensorDef, ranges: &[Range]) -> u64 {
    let ext_size: u64 = def.ext_indices.iter()
        .map(|idx| ranges[idx.range].size)
        .product();

    def.terms.iter().map(|term| {
        let sum_size: u64 = term.sum_indices.iter()
            .map(|idx| ranges[idx.range].size)
            .product();

        let contraction = if sum_size == 1 {
            ext_size
        } else {
            2 * ext_size * sum_size
        };
        contraction + ext_size  // addition into output
    }).sum()
}

/// Total FLOP cost of an entire computation.
fn total_cost(comp: &TensorComputation) -> u64 {
    comp.definitions.iter()
        .map(|def| def_cost(def, &comp.ranges))
        .sum()
}
```

### Canonicalization

Produces a canonical form for terms so that equivalent expressions (under dummy renaming and tensor symmetry) compare equal. Used to deduplicate intermediates.

**Algorithm:**

1. For each factor, apply symmetry generators to find the lexicographically smallest index arrangement, accumulating sign/conjugation actions on the coefficient.
2. Sort factors by `(tensor_id, canonical_indices)`.
3. Rename dummy (summed) indices by order of first appearance within each range: first dummy in range `r` becomes canonical ID 0, second becomes 1, etc.
4. Result: `CanonTerm { coeff: Rational, factors: Vec<CanonFactor> }` — hashable and comparable.

This reimplements drudge's `term.canon(symms=...)` over integer-ID structures.

## Serialization

All types derive `serde::Serialize` and `serde::Deserialize` for JSON interop. The Rust builder API is the primary interface:

```rust
let mut comp = TensorComputation::new();
let occ = comp.add_range("occ", 10);
let virt = comp.add_range("virt", 100);
let t = comp.add_tensor("t", &[occ, virt], vec![]);
let a = comp.add_tensor("A", &[occ, occ], vec![sym_generator]);
// ... define terms and definitions
```

## What This Design Does NOT Cover

- Optimization algorithms (greedy, MCTS, beam search, etc.)
- Transformation operations (biclique factorization, parenthesization)
- State management for search (copy-on-write `Rc` wrapping)

These will be designed separately as consumers of `TensorComputation`.
