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
