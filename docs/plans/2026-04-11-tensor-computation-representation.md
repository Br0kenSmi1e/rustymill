# Tensor Computation Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the core `TensorComputation` data types, cost model, canonicalization, builder API, and JSON serialization for the rustymill project.

**Architecture:** A Rust library crate with ID-based typed references (newtypes over `u32`), rational coefficients via the `num` crate, and serde for JSON. All types are immutable value types with `Clone`, `Debug`, `PartialEq`, `Eq`, `Hash`. Canonicalization operates over integer-ID structures using permutation group generators.

**Tech Stack:** Rust (edition 2021), `num` crate (rational arithmetic), `serde` + `serde_json` (serialization)

**Spec:** `docs/specs/2026-04-11-tensor-computation-representation-design.md`

**File Structure:**
```
~/rcode/rustymill/
├── Cargo.toml
├── src/
│   ├── lib.rs          # Module declarations + public re-exports
│   ├── repr.rs         # All core types: IDs, symmetry, tensors, TensorComputation builder
│   ├── cost.rs         # def_cost, total_cost
│   └── canon.rs        # Canonicalization (CanonTerm, CanonFactor, canon_term)
├── tests/
│   ├── repr_test.rs    # Tests for all core types and builder API
│   ├── cost_test.rs
│   ├── canon_test.rs
│   └── serde_test.rs
└── docs/
```

---

### Task 1: Initialize Rust Project

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`

- [ ] **Step 1: Initialize cargo project**

Run:
```bash
cd ~/rcode/rustymill && cargo init --lib
```

- [ ] **Step 2: Add dependencies to Cargo.toml**

Replace `Cargo.toml` contents with:

```toml
[package]
name = "rustymill"
version = "0.1.0"
edition = "2021"
description = "Tensor contraction optimizer"

[dependencies]
num = { version = "0.4", features = ["serde"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

- [ ] **Step 3: Set up lib.rs with module declarations**

Replace `src/lib.rs` with:

```rust
pub mod repr;
pub mod cost;
pub mod canon;
```

- [ ] **Step 4: Create empty module files**

`src/repr.rs`:
```rust
// Problem representation: IDs, symmetry, tensors, TensorComputation.
```

`src/cost.rs`:
```rust
// FLOP cost model: def_cost, total_cost.
```

`src/canon.rs`:
```rust
// Term canonicalization under dummy renaming and tensor symmetry.
```

- [ ] **Step 5: Verify it compiles**

Run:
```bash
cd ~/rcode/rustymill && cargo build
```
Expected: compiles with no errors.

- [ ] **Step 6: Initialize git and commit**

Run:
```bash
cd ~/rcode/rustymill && git init && git add -A && git commit -m "feat: initialize rustymill project skeleton"
```

---

### Task 2: Core Representation Types and Builder API

**Files:**
- Modify: `src/repr.rs`
- Create: `tests/repr_test.rs`

- [ ] **Step 1: Write failing tests**

Create `tests/repr_test.rs`:

```rust
use rustymill::repr::*;
use num::rational::Ratio;
use std::collections::HashSet;

// --- ID newtypes ---

#[test]
fn test_range_id_equality() {
    let a = RangeId(0);
    let b = RangeId(0);
    let c = RangeId(1);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_ids_are_hashable() {
    let mut set = HashSet::new();
    set.insert(RangeId(0));
    set.insert(RangeId(0));
    set.insert(RangeId(1));
    assert_eq!(set.len(), 2);
}

#[test]
fn test_ids_are_copy() {
    let a = RangeId(0);
    let b = a;
    assert_eq!(a, b);
}

// --- Symmetry ---

#[test]
fn test_sym_action_combine_identity() {
    assert_eq!(SymAction::Identity.combine(SymAction::Negate), SymAction::Negate);
}

#[test]
fn test_sym_action_combine_negate_negate() {
    assert_eq!(SymAction::Negate.combine(SymAction::Negate), SymAction::Identity);
}

#[test]
fn test_sym_action_combine_negate_conjugate() {
    assert_eq!(SymAction::Negate.combine(SymAction::Conjugate), SymAction::NegateConjugate);
}

#[test]
fn test_sym_action_combine_conjugate_conjugate() {
    assert_eq!(SymAction::Conjugate.combine(SymAction::Conjugate), SymAction::Identity);
}

#[test]
fn test_sym_action_combine_negate_conjugate_negate() {
    assert_eq!(
        SymAction::NegateConjugate.combine(SymAction::Negate),
        SymAction::Conjugate
    );
}

#[test]
fn test_sym_generator_apply() {
    let gen = SymGenerator {
        perm: vec![1, 0],
        action: SymAction::Negate,
    };
    let indices = vec![10u32, 20u32];
    let (permuted, action) = gen.apply(&indices);
    assert_eq!(permuted, vec![20, 10]);
    assert_eq!(action, SymAction::Negate);
}

#[test]
fn test_sym_generator_identity_perm() {
    let gen = SymGenerator {
        perm: vec![0, 1, 2],
        action: SymAction::Identity,
    };
    let indices = vec![5u32, 10u32, 15u32];
    let (permuted, action) = gen.apply(&indices);
    assert_eq!(permuted, vec![5, 10, 15]);
    assert_eq!(action, SymAction::Identity);
}

// --- Tensor data structures ---

#[test]
fn test_range_creation() {
    let r = Range { id: RangeId(0), size: 10 };
    assert_eq!(r.size, 10);
    assert_eq!(r.id, RangeId(0));
}

#[test]
fn test_index_creation() {
    let idx = Index { id: IndexId(0), range: RangeId(1) };
    assert_eq!(idx.id, IndexId(0));
    assert_eq!(idx.range, RangeId(1));
}

#[test]
fn test_tensor_info_with_symmetry() {
    let t = TensorInfo {
        id: TensorId(0),
        slots: vec![RangeId(0), RangeId(0)],
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Negate,
        }],
    };
    assert_eq!(t.symmetry.len(), 1);
}

#[test]
fn test_term_creation() {
    let term = Term {
        coeff: Ratio::new(3, 4),
        sum_indices: vec![Index { id: IndexId(2), range: RangeId(0) }],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(2)] },
            Factor { tensor: TensorId(1), indices: vec![IndexId(2), IndexId(1)] },
        ],
    };
    assert_eq!(*term.coeff.numer(), 3);
    assert_eq!(*term.coeff.denom(), 4);
    assert_eq!(term.sum_indices.len(), 1);
    assert_eq!(term.factors.len(), 2);
}

#[test]
fn test_terms_different_sum_indices() {
    let term1 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: IndexId(2), range: RangeId(0) }],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(2)] },
            Factor { tensor: TensorId(1), indices: vec![IndexId(2), IndexId(1)] },
        ],
    };
    let term2 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![
            Index { id: IndexId(3), range: RangeId(0) },
            Index { id: IndexId(4), range: RangeId(1) },
        ],
        factors: vec![
            Factor { tensor: TensorId(2), indices: vec![IndexId(0), IndexId(3), IndexId(4)] },
            Factor { tensor: TensorId(3), indices: vec![IndexId(3), IndexId(4), IndexId(1)] },
        ],
    };
    let def = TensorDef {
        base: TensorId(4),
        ext_indices: vec![
            Index { id: IndexId(0), range: RangeId(0) },
            Index { id: IndexId(1), range: RangeId(1) },
        ],
        terms: vec![term1, term2],
    };
    assert_eq!(def.terms[0].sum_indices.len(), 1);
    assert_eq!(def.terms[1].sum_indices.len(), 2);
}

// --- TensorComputation builder ---

#[test]
fn test_new_computation_is_empty() {
    let comp = TensorComputation::new();
    assert!(comp.ranges().is_empty());
    assert!(comp.tensors().is_empty());
    assert!(comp.definitions().is_empty());
}

#[test]
fn test_add_range() {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let virt = comp.add_range(100);
    assert_eq!(occ, RangeId(0));
    assert_eq!(virt, RangeId(1));
    assert_eq!(comp.ranges().len(), 2);
    assert_eq!(comp.ranges()[0].size, 10);
    assert_eq!(comp.ranges()[1].size, 100);
}

#[test]
fn test_add_tensor() {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let virt = comp.add_range(100);
    let t = comp.add_tensor(&[occ, virt], vec![]);
    assert_eq!(t, TensorId(0));
    assert_eq!(comp.tensors().len(), 1);
    assert_eq!(comp.tensors()[0].slots, vec![occ, virt]);
}

#[test]
fn test_add_tensor_with_symmetry() {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let sym = SymGenerator { perm: vec![1, 0], action: SymAction::Negate };
    let v = comp.add_tensor(&[occ, occ], vec![sym.clone()]);
    assert_eq!(comp.tensors()[v.0 as usize].symmetry, vec![sym]);
}

#[test]
fn test_add_definition() {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let virt = comp.add_range(100);
    let a = comp.add_tensor(&[occ, virt], vec![]);
    let b = comp.add_tensor(&[virt, occ], vec![]);
    let t = comp.add_tensor(&[occ, occ], vec![]);

    let i = IndexId(0);
    let j = IndexId(1);
    let k = IndexId(2);

    comp.add_definition(
        t,
        vec![
            Index { id: i, range: occ },
            Index { id: j, range: occ },
        ],
        vec![Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index { id: k, range: virt }],
            factors: vec![
                Factor { tensor: a, indices: vec![i, k] },
                Factor { tensor: b, indices: vec![k, j] },
            ],
        }],
    );

    assert_eq!(comp.definitions().len(), 1);
    assert_eq!(comp.definitions()[0].base, t);
    assert_eq!(comp.definitions()[0].ext_indices.len(), 2);
    assert_eq!(comp.definitions()[0].terms.len(), 1);
}

#[test]
fn test_full_computation() {
    // r[a,b] = A[a,c]*B[c,b] + C[a,d,e]*D[d,e,b]
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let virt = comp.add_range(100);

    let a_tensor = comp.add_tensor(&[occ, occ], vec![]);
    let b_tensor = comp.add_tensor(&[occ, occ], vec![]);
    let c_tensor = comp.add_tensor(&[occ, virt, virt], vec![]);
    let d_tensor = comp.add_tensor(&[virt, virt, occ], vec![]);
    let r_tensor = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);
    let d = IndexId(3);
    let e = IndexId(4);

    comp.add_definition(
        r_tensor,
        vec![
            Index { id: a, range: occ },
            Index { id: b, range: occ },
        ],
        vec![
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![Index { id: c, range: occ }],
                factors: vec![
                    Factor { tensor: a_tensor, indices: vec![a, c] },
                    Factor { tensor: b_tensor, indices: vec![c, b] },
                ],
            },
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![
                    Index { id: d, range: virt },
                    Index { id: e, range: virt },
                ],
                factors: vec![
                    Factor { tensor: c_tensor, indices: vec![a, d, e] },
                    Factor { tensor: d_tensor, indices: vec![d, e, b] },
                ],
            },
        ],
    );

    assert_eq!(comp.tensors().len(), 5);
    assert_eq!(comp.definitions().len(), 1);
    assert_eq!(comp.definitions()[0].terms.len(), 2);
    assert_eq!(comp.definitions()[0].terms[0].sum_indices.len(), 1);
    assert_eq!(comp.definitions()[0].terms[1].sum_indices.len(), 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd ~/rcode/rustymill && cargo test --test repr_test
```
Expected: FAIL — `repr` module not implemented.

- [ ] **Step 3: Implement repr.rs**

Replace `src/repr.rs` with:

```rust
use num::rational::Ratio;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ID newtypes
// ---------------------------------------------------------------------------

/// Identifies a Range (index space with known dimension).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RangeId(pub u32);

/// Identifies an Index variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IndexId(pub u32);

/// Identifies a Tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TensorId(pub u32);

// ---------------------------------------------------------------------------
// Symmetry
// ---------------------------------------------------------------------------

/// Action on coefficient when permuting tensor index slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymAction {
    /// No change.
    Identity,
    /// Multiply coefficient by -1.
    Negate,
    /// Complex-conjugate the coefficient.
    Conjugate,
    /// Both negate and conjugate.
    NegateConjugate,
}

impl SymAction {
    /// Compose two actions: self followed by other.
    pub fn combine(self, other: SymAction) -> SymAction {
        let (sn, sc) = self.to_bits();
        let (on, oc) = other.to_bits();
        Self::from_bits(sn ^ on, sc ^ oc)
    }

    fn to_bits(self) -> (bool, bool) {
        match self {
            SymAction::Identity => (false, false),
            SymAction::Negate => (true, false),
            SymAction::Conjugate => (false, true),
            SymAction::NegateConjugate => (true, true),
        }
    }

    fn from_bits(negate: bool, conjugate: bool) -> SymAction {
        match (negate, conjugate) {
            (false, false) => SymAction::Identity,
            (true, false) => SymAction::Negate,
            (false, true) => SymAction::Conjugate,
            (true, true) => SymAction::NegateConjugate,
        }
    }
}

/// A generator of a tensor's symmetry group: a permutation of index slots
/// paired with an action on the coefficient.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymGenerator {
    /// Permutation of index slots. `perm[i]` is the slot that slot `i` maps to.
    pub perm: Vec<usize>,
    /// Action applied to the coefficient under this permutation.
    pub action: SymAction,
}

impl SymGenerator {
    /// Apply this generator to a slice of index values.
    /// Returns the permuted indices and the action on the coefficient.
    pub fn apply<T: Copy>(&self, indices: &[T]) -> (Vec<T>, SymAction) {
        assert_eq!(self.perm.len(), indices.len());
        let permuted = self.perm.iter().map(|&p| indices[p]).collect();
        (permuted, self.action)
    }
}

// ---------------------------------------------------------------------------
// Tensor data structures
// ---------------------------------------------------------------------------

/// A rational coefficient (i64 numerator and denominator).
pub type Rational = Ratio<i64>;

/// A named index space with a known dimension.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Range {
    pub id: RangeId,
    pub size: u64,
}

/// An index variable scoped to a specific range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Index {
    pub id: IndexId,
    pub range: RangeId,
}

/// A named tensor with index slot structure and symmetry group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    pub id: TensorId,
    /// Each slot expects indices from this range.
    pub slots: Vec<RangeId>,
    /// Generators of the symmetry group acting on index slots.
    pub symmetry: Vec<SymGenerator>,
}

/// A tensor applied to concrete indices (one factor in a product).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Factor {
    pub tensor: TensorId,
    pub indices: Vec<IndexId>,
}

/// A rational coefficient times a product of factors, with its own summation indices.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Term {
    pub coeff: Rational,
    /// Summation (dummy) indices for this term.
    pub sum_indices: Vec<Index>,
    pub factors: Vec<Factor>,
}

/// A tensor definition: LHS[ext_indices] = term1 + term2 + ...
///
/// External indices are shared across all terms (they must match the LHS).
/// Each term brings its own summation indices.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDef {
    pub base: TensorId,
    pub ext_indices: Vec<Index>,
    pub terms: Vec<Term>,
}

// ---------------------------------------------------------------------------
// TensorComputation container and builder API
// ---------------------------------------------------------------------------

/// The top-level container for a tensor computation.
/// Both input and output of any optimization algorithm.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorComputation {
    ranges: Vec<Range>,
    tensors: Vec<TensorInfo>,
    definitions: Vec<TensorDef>,
}

impl TensorComputation {
    /// Create an empty tensor computation.
    pub fn new() -> Self {
        TensorComputation {
            ranges: Vec::new(),
            tensors: Vec::new(),
            definitions: Vec::new(),
        }
    }

    /// Add a range (index space) with the given dimension size. Returns its ID.
    pub fn add_range(&mut self, size: u64) -> RangeId {
        let id = RangeId(self.ranges.len() as u32);
        self.ranges.push(Range { id, size });
        id
    }

    /// Add a tensor with the given index slot ranges and symmetry generators.
    /// Returns its ID.
    pub fn add_tensor(&mut self, slots: &[RangeId], symmetry: Vec<SymGenerator>) -> TensorId {
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorInfo {
            id,
            slots: slots.to_vec(),
            symmetry,
        });
        id
    }

    /// Add a tensor definition.
    pub fn add_definition(
        &mut self,
        base: TensorId,
        ext_indices: Vec<Index>,
        terms: Vec<Term>,
    ) {
        self.definitions.push(TensorDef {
            base,
            ext_indices,
            terms,
        });
    }

    /// Access the ranges.
    pub fn ranges(&self) -> &[Range] {
        &self.ranges
    }

    /// Access the tensors.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Access the definitions.
    pub fn definitions(&self) -> &[TensorDef] {
        &self.definitions
    }
}

impl Default for TensorComputation {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd ~/rcode/rustymill && cargo test --test repr_test
```
Expected: all 21 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/rcode/rustymill && git add src/repr.rs tests/repr_test.rs && git commit -m "feat: add core representation types and TensorComputation builder"
```

---

### Task 3: Cost Model

**Files:**
- Modify: `src/cost.rs`
- Create: `tests/cost_test.rs`

- [ ] **Step 1: Write failing test for cost model**

Create `tests/cost_test.rs`:

```rust
use rustymill::repr::*;
use rustymill::cost::{def_cost, total_cost};
use num::rational::Ratio;

/// Helper: build a simple computation with one definition.
fn simple_contraction() -> TensorComputation {
    // t[a,b] = sum_c A[a,c] * B[c,b]
    // occ=10 for all indices
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _a = comp.add_tensor(&[occ, occ], vec![]);
    let _b = comp.add_tensor(&[occ, occ], vec![]);
    let t = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);

    comp.add_definition(
        t,
        vec![
            Index { id: a, range: occ },
            Index { id: b, range: occ },
        ],
        vec![Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index { id: c, range: occ }],
            factors: vec![
                Factor { tensor: TensorId(0), indices: vec![a, c] },
                Factor { tensor: TensorId(1), indices: vec![c, b] },
            ],
        }],
    );
    comp
}

#[test]
fn test_def_cost_simple_contraction() {
    let comp = simple_contraction();
    // ext_size = 10 * 10 = 100, sum_size = 10
    // cost = 2 * 100 * 10 + 100 = 2100
    let cost = def_cost(&comp.definitions()[0], comp.ranges());
    assert_eq!(cost, 2100);
}

#[test]
fn test_def_cost_no_summation() {
    // t[a,b] = A[a,b]  (just a copy, no sum indices)
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _a = comp.add_tensor(&[occ, occ], vec![]);
    let t = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);

    comp.add_definition(
        t,
        vec![
            Index { id: a, range: occ },
            Index { id: b, range: occ },
        ],
        vec![Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![],
            factors: vec![
                Factor { tensor: TensorId(0), indices: vec![a, b] },
            ],
        }],
    );

    // ext_size = 100, sum_size = 1 (empty product)
    // cost = 100 + 100 = 200
    let cost = def_cost(&comp.definitions()[0], comp.ranges());
    assert_eq!(cost, 200);
}

#[test]
fn test_def_cost_two_terms_different_sums() {
    // t[a,b] = A[a,c]*B[c,b] + C[a,d,e]*D[d,e,b]
    // occ=10, virt=100
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let virt = comp.add_range(100);
    let _a = comp.add_tensor(&[occ, occ], vec![]);
    let _b = comp.add_tensor(&[occ, occ], vec![]);
    let _c = comp.add_tensor(&[occ, virt, virt], vec![]);
    let _d = comp.add_tensor(&[virt, virt, occ], vec![]);
    let t = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);
    let d = IndexId(3);
    let e = IndexId(4);

    comp.add_definition(
        t,
        vec![
            Index { id: a, range: occ },
            Index { id: b, range: occ },
        ],
        vec![
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![Index { id: c, range: occ }],
                factors: vec![
                    Factor { tensor: TensorId(0), indices: vec![a, c] },
                    Factor { tensor: TensorId(1), indices: vec![c, b] },
                ],
            },
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![
                    Index { id: d, range: virt },
                    Index { id: e, range: virt },
                ],
                factors: vec![
                    Factor { tensor: TensorId(2), indices: vec![a, d, e] },
                    Factor { tensor: TensorId(3), indices: vec![d, e, b] },
                ],
            },
        ],
    );

    // ext_size = 10 * 10 = 100
    // term1: sum_size = 10, cost = 2 * 100 * 10 + 100 = 2100
    // term2: sum_size = 100 * 100 = 10000, cost = 2 * 100 * 10000 + 100 = 2_000_100
    // total = 2100 + 2_000_100 = 2_002_200
    let cost = def_cost(&comp.definitions()[0], comp.ranges());
    assert_eq!(cost, 2_002_200);
}

#[test]
fn test_total_cost_multiple_definitions() {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _a = comp.add_tensor(&[occ, occ], vec![]);
    let _b = comp.add_tensor(&[occ, occ], vec![]);
    let t1 = comp.add_tensor(&[occ, occ], vec![]);
    let t2 = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);

    let terms = vec![Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: c, range: occ }],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a, c] },
            Factor { tensor: TensorId(1), indices: vec![c, b] },
        ],
    }];

    comp.add_definition(
        t1,
        vec![
            Index { id: a, range: occ },
            Index { id: b, range: occ },
        ],
        terms.clone(),
    );
    comp.add_definition(
        t2,
        vec![
            Index { id: a, range: occ },
            Index { id: b, range: occ },
        ],
        terms,
    );

    // Each def costs 2100, total = 4200
    let cost = total_cost(&comp);
    assert_eq!(cost, 4200);
}

#[test]
fn test_def_cost_scalar_output() {
    // E = sum_{a,b} A[a,b] * B[a,b]  (scalar, no ext indices)
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _a = comp.add_tensor(&[occ, occ], vec![]);
    let _b = comp.add_tensor(&[occ, occ], vec![]);
    let e = comp.add_tensor(&[], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);

    comp.add_definition(
        e,
        vec![],
        vec![Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![
                Index { id: a, range: occ },
                Index { id: b, range: occ },
            ],
            factors: vec![
                Factor { tensor: TensorId(0), indices: vec![a, b] },
                Factor { tensor: TensorId(1), indices: vec![a, b] },
            ],
        }],
    );

    // ext_size = 1 (empty product), sum_size = 10 * 10 = 100
    // cost = 2 * 1 * 100 + 1 = 201
    let cost = def_cost(&comp.definitions()[0], comp.ranges());
    assert_eq!(cost, 201);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd ~/rcode/rustymill && cargo test --test cost_test
```
Expected: FAIL — `def_cost` and `total_cost` not defined.

- [ ] **Step 3: Implement cost model**

Replace `src/cost.rs` with:

```rust
use crate::repr::{Range, TensorComputation, TensorDef};

/// FLOP cost of evaluating one TensorDef.
///
/// For each term: contraction cost + addition into output.
/// - No summation: ext_size (copy/scale) + ext_size (addition)
/// - With summation: 2 * ext_size * sum_size (multiply-add) + ext_size (addition)
pub fn def_cost(def: &TensorDef, ranges: &[Range]) -> u64 {
    let ext_size: u64 = def
        .ext_indices
        .iter()
        .map(|idx| ranges[idx.range.0 as usize].size)
        .product::<u64>()
        .max(1); // scalar output: empty product = 1

    def.terms
        .iter()
        .map(|term| {
            let sum_size: u64 = term
                .sum_indices
                .iter()
                .map(|idx| ranges[idx.range.0 as usize].size)
                .product::<u64>()
                .max(1); // no summation: empty product = 1

            let contraction = if sum_size == 1 {
                ext_size
            } else {
                2 * ext_size * sum_size
            };
            contraction + ext_size
        })
        .sum()
}

/// Total FLOP cost of an entire computation.
pub fn total_cost(comp: &TensorComputation) -> u64 {
    comp.definitions()
        .iter()
        .map(|def| def_cost(def, comp.ranges()))
        .sum()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd ~/rcode/rustymill && cargo test --test cost_test
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/rcode/rustymill && git add src/cost.rs tests/cost_test.rs && git commit -m "feat: add FLOP cost model (def_cost, total_cost)"
```

---

### Task 4: Canonicalization

**Files:**
- Modify: `src/canon.rs`
- Create: `tests/canon_test.rs`

- [ ] **Step 1: Write failing test for canonicalization**

Create `tests/canon_test.rs`:

```rust
use rustymill::canon::canon_term;
use rustymill::repr::*;
use num::rational::Ratio;

fn make_no_sym_tensor(id: TensorId, slots: &[RangeId]) -> TensorInfo {
    TensorInfo {
        id,
        slots: slots.to_vec(),
        symmetry: vec![],
    }
}

fn make_antisym_tensor(id: TensorId, range: RangeId) -> TensorInfo {
    TensorInfo {
        id,
        slots: vec![range, range],
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Negate,
        }],
    }
}

#[test]
fn test_canon_sorts_factors() {
    // Term: B[c,b] * A[a,c] should canonicalize to A[...] * B[...]
    let occ = RangeId(0);
    let tensors = vec![
        make_no_sym_tensor(TensorId(0), &[occ, occ]),
        make_no_sym_tensor(TensorId(1), &[occ, occ]),
    ];

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);

    let term1 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: c, range: occ }],
        factors: vec![
            Factor { tensor: TensorId(1), indices: vec![c, b] },
            Factor { tensor: TensorId(0), indices: vec![a, c] },
        ],
    };

    let term2 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: c, range: occ }],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a, c] },
            Factor { tensor: TensorId(1), indices: vec![c, b] },
        ],
    };

    let ext = vec![
        Index { id: a, range: occ },
        Index { id: b, range: occ },
    ];

    let c1 = canon_term(&term1, &ext, &tensors);
    let c2 = canon_term(&term2, &ext, &tensors);
    assert_eq!(c1, c2);
}

#[test]
fn test_canon_dummy_renaming() {
    // A[a,c]*B[c,b] with c as dummy should equal A[a,d]*B[d,b] with d as dummy
    let occ = RangeId(0);
    let tensors = vec![
        make_no_sym_tensor(TensorId(0), &[occ, occ]),
        make_no_sym_tensor(TensorId(1), &[occ, occ]),
    ];

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);
    let d = IndexId(3);

    let term1 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: c, range: occ }],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a, c] },
            Factor { tensor: TensorId(1), indices: vec![c, b] },
        ],
    };

    let term2 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: d, range: occ }],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a, d] },
            Factor { tensor: TensorId(1), indices: vec![d, b] },
        ],
    };

    let ext = vec![
        Index { id: a, range: occ },
        Index { id: b, range: occ },
    ];

    let c1 = canon_term(&term1, &ext, &tensors);
    let c2 = canon_term(&term2, &ext, &tensors);
    assert_eq!(c1, c2);
}

#[test]
fn test_canon_antisymmetric_tensor() {
    // V[i,j] antisymmetric: V[j,i] = -V[i,j]
    let occ = RangeId(0);
    let tensors = vec![make_antisym_tensor(TensorId(0), occ)];

    let i = IndexId(0);
    let j = IndexId(1);

    let term_ji = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![],
        factors: vec![Factor { tensor: TensorId(0), indices: vec![j, i] }],
    };

    let term_ij = Term {
        coeff: Ratio::new(-1, 1),
        sum_indices: vec![],
        factors: vec![Factor { tensor: TensorId(0), indices: vec![i, j] }],
    };

    let ext = vec![
        Index { id: i, range: occ },
        Index { id: j, range: occ },
    ];

    let c1 = canon_term(&term_ji, &ext, &tensors);
    let c2 = canon_term(&term_ij, &ext, &tensors);
    assert_eq!(c1, c2);
}

#[test]
fn test_canon_symmetric_tensor() {
    let occ = RangeId(0);
    let tensors = vec![TensorInfo {
        id: TensorId(0),
        slots: vec![occ, occ],
        symmetry: vec![SymGenerator {
            perm: vec![1, 0],
            action: SymAction::Identity,
        }],
    }];

    let i = IndexId(0);
    let j = IndexId(1);

    let term_ji = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![],
        factors: vec![Factor { tensor: TensorId(0), indices: vec![j, i] }],
    };

    let term_ij = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![],
        factors: vec![Factor { tensor: TensorId(0), indices: vec![i, j] }],
    };

    let ext = vec![
        Index { id: i, range: occ },
        Index { id: j, range: occ },
    ];

    let c1 = canon_term(&term_ji, &ext, &tensors);
    let c2 = canon_term(&term_ij, &ext, &tensors);
    assert_eq!(c1, c2);
    assert_eq!(*c1.coeff.numer(), 1);
}

#[test]
fn test_canon_dummy_renaming_across_ranges() {
    // A[a,c]*B[c,d]*C[d,b] where c:occ, d:virt
    // vs A[a,e]*B[e,f]*C[f,b] where e:occ, f:virt
    let occ = RangeId(0);
    let virt = RangeId(1);
    let tensors = vec![
        make_no_sym_tensor(TensorId(0), &[occ, occ]),
        make_no_sym_tensor(TensorId(1), &[occ, virt]),
        make_no_sym_tensor(TensorId(2), &[virt, occ]),
    ];

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);
    let d = IndexId(3);
    let e = IndexId(4);
    let f = IndexId(5);

    let ext = vec![
        Index { id: a, range: occ },
        Index { id: b, range: occ },
    ];

    let term1 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![
            Index { id: c, range: occ },
            Index { id: d, range: virt },
        ],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a, c] },
            Factor { tensor: TensorId(1), indices: vec![c, d] },
            Factor { tensor: TensorId(2), indices: vec![d, b] },
        ],
    };

    let term2 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![
            Index { id: e, range: occ },
            Index { id: f, range: virt },
        ],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a, e] },
            Factor { tensor: TensorId(1), indices: vec![e, f] },
            Factor { tensor: TensorId(2), indices: vec![f, b] },
        ],
    };

    let c1 = canon_term(&term1, &ext, &tensors);
    let c2 = canon_term(&term2, &ext, &tensors);
    assert_eq!(c1, c2);
}

#[test]
fn test_canon_different_coefficients() {
    let occ = RangeId(0);
    let tensors = vec![make_no_sym_tensor(TensorId(0), &[occ, occ])];
    let i = IndexId(0);
    let j = IndexId(1);
    let ext = vec![
        Index { id: i, range: occ },
        Index { id: j, range: occ },
    ];

    let term1 = Term {
        coeff: Ratio::new(3, 4),
        sum_indices: vec![],
        factors: vec![Factor { tensor: TensorId(0), indices: vec![i, j] }],
    };
    let term2 = Term {
        coeff: Ratio::new(1, 2),
        sum_indices: vec![],
        factors: vec![Factor { tensor: TensorId(0), indices: vec![i, j] }],
    };

    let c1 = canon_term(&term1, &ext, &tensors);
    let c2 = canon_term(&term2, &ext, &tensors);
    assert_ne!(c1, c2);
    assert_eq!(c1.coeff, Ratio::new(3, 4));
    assert_eq!(c2.coeff, Ratio::new(1, 2));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd ~/rcode/rustymill && cargo test --test canon_test
```
Expected: FAIL — `canon` module not implemented.

- [ ] **Step 3: Implement canonicalization**

Replace `src/canon.rs` with:

```rust
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::repr::{
    IndexId, Rational, RangeId, SymAction, SymGenerator, Index, TensorId, TensorInfo, Term,
};

/// A canonicalized factor: tensor ID with canonical index slots.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CanonFactor {
    pub tensor: TensorId,
    pub indices: Vec<CanonIndex>,
}

/// A canonical index: either an external index (kept as-is) or a
/// dummy index (renumbered per range).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CanonIndex {
    pub range: RangeId,
    /// External indices keep their original IndexId values.
    /// Dummy indices are renumbered starting from u32::MAX downward
    /// to avoid collisions with external IDs.
    pub canon_id: u32,
}

/// A canonicalized term: comparable and hashable.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonTerm {
    pub coeff: Rational,
    pub factors: Vec<CanonFactor>,
}

/// Canonicalize a term under tensor symmetries and dummy renaming.
///
/// 1. For each factor, apply all symmetry generators and pick the
///    lexicographically smallest index arrangement, accumulating the
///    action on the coefficient.
/// 2. Sort factors by (tensor_id, canonical_indices).
/// 3. Rename dummy indices by order of first appearance within each range.
pub fn canon_term(term: &Term, ext_indices: &[Index], tensors: &[TensorInfo]) -> CanonTerm {
    let ext_set: HashMap<IndexId, RangeId> = ext_indices
        .iter()
        .map(|idx| (idx.id, idx.range))
        .collect();
    let dummy_set: HashMap<IndexId, RangeId> = term
        .sum_indices
        .iter()
        .map(|idx| (idx.id, idx.range))
        .collect();

    // Step 1: Canonicalize each factor under its tensor's symmetry group.
    let mut coeff_action = SymAction::Identity;
    let mut raw_factors: Vec<(TensorId, Vec<IndexId>)> = Vec::with_capacity(term.factors.len());

    for factor in &term.factors {
        let info = &tensors[factor.tensor.0 as usize];
        let (best_indices, action) = canon_factor_indices(&factor.indices, &info.symmetry);
        coeff_action = coeff_action.combine(action);
        raw_factors.push((factor.tensor, best_indices));
    }

    // Step 2: Sort factors by (tensor_id, indices).
    raw_factors.sort();

    // Step 3: Rename dummy indices by first appearance per range.
    let mut dummy_canon: HashMap<IndexId, u32> = HashMap::new();
    let mut range_counters: HashMap<RangeId, u32> = HashMap::new();

    for (_, indices) in &raw_factors {
        for &idx in indices {
            if dummy_set.contains_key(&idx) && !dummy_canon.contains_key(&idx) {
                let range = dummy_set[&idx];
                let counter = range_counters.entry(range).or_insert(0);
                dummy_canon.insert(idx, u32::MAX - *counter);
                *counter += 1;
            }
        }
    }

    // Build canonical factors.
    let canon_factors: Vec<CanonFactor> = raw_factors
        .into_iter()
        .map(|(tensor, indices)| {
            let canon_indices = indices
                .into_iter()
                .map(|idx| {
                    if let Some(&range) = ext_set.get(&idx) {
                        CanonIndex {
                            range,
                            canon_id: idx.0,
                        }
                    } else {
                        let range = dummy_set[&idx];
                        CanonIndex {
                            range,
                            canon_id: dummy_canon[&idx],
                        }
                    }
                })
                .collect();
            CanonFactor {
                tensor,
                indices: canon_indices,
            }
        })
        .collect();

    // Apply accumulated symmetry action to the coefficient.
    let coeff = apply_action_to_coeff(term.coeff, coeff_action);

    CanonTerm {
        coeff,
        factors: canon_factors,
    }
}

/// Given a factor's indices and its tensor's symmetry generators, find the
/// lexicographically smallest permutation and the accumulated action.
///
/// Uses brute-force enumeration of the group generated by the generators.
/// For typical tensor symmetries (2-4 index slots), the group is small.
fn canon_factor_indices(
    indices: &[IndexId],
    generators: &[SymGenerator],
) -> (Vec<IndexId>, SymAction) {
    if generators.is_empty() {
        return (indices.to_vec(), SymAction::Identity);
    }

    // Generate all group elements by BFS from identity.
    let n = indices.len();
    let identity_perm: Vec<usize> = (0..n).collect();

    let mut elements: Vec<(Vec<usize>, SymAction)> = vec![(identity_perm, SymAction::Identity)];
    let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
    seen.insert((0..n).collect());

    let mut queue_idx = 0;
    while queue_idx < elements.len() {
        let (perm, action) = elements[queue_idx].clone();
        queue_idx += 1;

        for gen in generators {
            // Compose: new_perm[i] = perm[gen.perm[i]]
            let new_perm: Vec<usize> = gen.perm.iter().map(|&g| perm[g]).collect();
            if seen.insert(new_perm.clone()) {
                let new_action = action.combine(gen.action);
                elements.push((new_perm, new_action));
            }
        }
    }

    // Find the permutation that produces the lexicographically smallest indices.
    let mut best_indices = indices.to_vec();
    let mut best_action = SymAction::Identity;

    for (perm, action) in &elements {
        let permuted: Vec<IndexId> = perm.iter().map(|&p| indices[p]).collect();
        if permuted < best_indices {
            best_indices = permuted;
            best_action = *action;
        }
    }

    (best_indices, best_action)
}

/// Apply a symmetry action to a rational coefficient.
fn apply_action_to_coeff(coeff: Rational, action: SymAction) -> Rational {
    match action {
        SymAction::Identity => coeff,
        SymAction::Negate => -coeff,
        SymAction::Conjugate => coeff,
        SymAction::NegateConjugate => -coeff,
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd ~/rcode/rustymill && cargo test --test canon_test
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/rcode/rustymill && git add src/canon.rs tests/canon_test.rs && git commit -m "feat: add term canonicalization under symmetry and dummy renaming"
```

---

### Task 5: JSON Serialization

**Files:**
- Create: `tests/serde_test.rs`

- [ ] **Step 1: Write test for JSON round-trip**

Create `tests/serde_test.rs`:

```rust
use rustymill::repr::*;
use num::rational::Ratio;

fn build_sample_computation() -> TensorComputation {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let virt = comp.add_range(100);

    let sym = SymGenerator { perm: vec![1, 0], action: SymAction::Negate };
    let v = comp.add_tensor(&[occ, occ, virt, virt], vec![sym]);
    let t = comp.add_tensor(&[occ, occ], vec![]);
    let r = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);
    let d = IndexId(3);

    comp.add_definition(
        r,
        vec![
            Index { id: a, range: occ },
            Index { id: b, range: occ },
        ],
        vec![Term {
            coeff: Ratio::new(1, 2),
            sum_indices: vec![
                Index { id: c, range: virt },
                Index { id: d, range: virt },
            ],
            factors: vec![
                Factor { tensor: v, indices: vec![a, b, c, d] },
                Factor { tensor: t, indices: vec![c, d] },
            ],
        }],
    );
    comp
}

#[test]
fn test_json_round_trip() {
    let comp = build_sample_computation();
    let json = serde_json::to_string_pretty(&comp).unwrap();
    let deserialized: TensorComputation = serde_json::from_str(&json).unwrap();
    assert_eq!(comp, deserialized);
}

#[test]
fn test_json_contains_expected_fields() {
    let comp = build_sample_computation();
    let json = serde_json::to_string(&comp).unwrap();
    assert!(json.contains("\"ranges\""));
    assert!(json.contains("\"tensors\""));
    assert!(json.contains("\"definitions\""));
    assert!(json.contains("\"Negate\""));
    assert!(json.contains("\"size\":10"));
    assert!(json.contains("\"size\":100"));
}

#[test]
fn test_json_empty_computation() {
    let comp = TensorComputation::new();
    let json = serde_json::to_string(&comp).unwrap();
    let deserialized: TensorComputation = serde_json::from_str(&json).unwrap();
    assert_eq!(comp, deserialized);
}
```

- [ ] **Step 2: Run tests**

Run:
```bash
cd ~/rcode/rustymill && cargo test --test serde_test
```
Expected: all 3 tests PASS (serde derives are already on all types, `num` serde feature enabled in Task 1).

- [ ] **Step 3: Commit**

```bash
cd ~/rcode/rustymill && git add tests/serde_test.rs && git commit -m "feat: add JSON serialization tests"
```

---

### Task 6: Public API Re-exports

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Update lib.rs with re-exports**

Replace `src/lib.rs` with:

```rust
pub mod repr;
pub mod cost;
pub mod canon;

// Re-export primary types at crate root for convenience.
pub use repr::TensorComputation;
pub use cost::{def_cost, total_cost};
pub use canon::{canon_term, CanonTerm, CanonFactor, CanonIndex};
```

- [ ] **Step 2: Verify everything compiles and all tests pass**

Run:
```bash
cd ~/rcode/rustymill && cargo test
```
Expected: all tests across all test files PASS.

- [ ] **Step 3: Commit**

```bash
cd ~/rcode/rustymill && git add src/lib.rs && git commit -m "feat: add public API re-exports"
```
