use rustymill::canon::canon_term;
use rustymill::repr::*;
use num::rational::Ratio;

fn make_no_sym_tensor(id: TensorId, slots: &[RangeId]) -> TensorInfo {
    TensorInfo { id, slots: slots.to_vec(), symmetry: vec![] }
}

fn make_antisym_tensor(id: TensorId, range: RangeId) -> TensorInfo {
    TensorInfo {
        id,
        slots: vec![range, range],
        symmetry: vec![SymGenerator { perm: vec![1, 0], action: SymAction::Negate }],
    }
}

#[test]
fn test_canon_sorts_factors() {
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
    let ext = vec![Index { id: a, range: occ }, Index { id: b, range: occ }];
    let c1 = canon_term(&term1, &ext, &tensors);
    let c2 = canon_term(&term2, &ext, &tensors);
    assert_eq!(c1, c2);
}

#[test]
fn test_canon_dummy_renaming() {
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
    let ext = vec![Index { id: a, range: occ }, Index { id: b, range: occ }];
    let c1 = canon_term(&term1, &ext, &tensors);
    let c2 = canon_term(&term2, &ext, &tensors);
    assert_eq!(c1, c2);
}

#[test]
fn test_canon_antisymmetric_tensor() {
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
    let ext = vec![Index { id: i, range: occ }, Index { id: j, range: occ }];
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
        symmetry: vec![SymGenerator { perm: vec![1, 0], action: SymAction::Identity }],
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
    let ext = vec![Index { id: i, range: occ }, Index { id: j, range: occ }];
    let c1 = canon_term(&term_ji, &ext, &tensors);
    let c2 = canon_term(&term_ij, &ext, &tensors);
    assert_eq!(c1, c2);
    assert_eq!(*c1.coeff.numer(), 1);
}

#[test]
fn test_canon_dummy_renaming_across_ranges() {
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
    let ext = vec![Index { id: a, range: occ }, Index { id: b, range: occ }];
    let term1 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: c, range: occ }, Index { id: d, range: virt }],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a, c] },
            Factor { tensor: TensorId(1), indices: vec![c, d] },
            Factor { tensor: TensorId(2), indices: vec![d, b] },
        ],
    };
    let term2 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: e, range: occ }, Index { id: f, range: virt }],
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
    let ext = vec![Index { id: i, range: occ }, Index { id: j, range: occ }];
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
