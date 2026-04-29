use num::rational::Ratio;
use rustymill::repr::*;
use rustymill::rl_canon::{
    build_canon_def_context, canon_split, canon_term, canonical_term_key,
};
use rustymill::rl_parenth::{LastStepIndices, TermSplit};

fn make_context_def_and_tensors() -> (TensorDef, Vec<TensorInfo>) {
    let def = TensorDef {
        base: TensorId(99),
        ext_indices: vec![
            Index { id: IndexId(0), range: RangeId(0) },
            Index { id: IndexId(1), range: RangeId(1) },
        ],
        terms: vec![
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![
                    Index { id: IndexId(2), range: RangeId(0) },
                    Index { id: IndexId(3), range: RangeId(1) },
                ],
                factors: vec![
                    Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(2)] },
                    Factor { tensor: TensorId(1), indices: vec![IndexId(2), IndexId(3)] },
                    Factor { tensor: TensorId(2), indices: vec![IndexId(3), IndexId(1)] },
                ],
            },
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![
                    Index { id: IndexId(5), range: RangeId(0) },
                    Index { id: IndexId(6), range: RangeId(1) },
                ],
                factors: vec![
                    Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(5)] },
                    Factor { tensor: TensorId(1), indices: vec![IndexId(5), IndexId(6)] },
                    Factor { tensor: TensorId(2), indices: vec![IndexId(6), IndexId(1)] },
                ],
            },
        ],
    };

    let tensors = vec![
        TensorInfo { id: TensorId(0), symmetry: vec![] },
        TensorInfo { id: TensorId(1), symmetry: vec![] },
        TensorInfo { id: TensorId(2), symmetry: vec![] },
    ];

    (def, tensors)
}

fn make_split() -> TermSplit {
    TermSplit {
        left_sub_term: Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index { id: IndexId(2), range: RangeId(0) }],
            factors: vec![Factor {
                tensor: TensorId(0),
                indices: vec![IndexId(0), IndexId(2)],
            }],
        },
        right_sub_term: Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![
                Index { id: IndexId(2), range: RangeId(0) },
                Index { id: IndexId(3), range: RangeId(1) },
            ],
            factors: vec![
                Factor { tensor: TensorId(1), indices: vec![IndexId(2), IndexId(3)] },
                Factor { tensor: TensorId(2), indices: vec![IndexId(3), IndexId(1)] },
            ],
        },
        last_step: LastStepIndices {
            left_ext: 1,
            right_ext: 2,
            sums: vec![RangeId(0)],
        },
    }
}

fn make_same_range_context_def_and_tensors() -> (TensorDef, Vec<TensorInfo>) {
    let def = TensorDef {
        base: TensorId(102),
        ext_indices: vec![
            Index { id: IndexId(0), range: RangeId(0) },
            Index { id: IndexId(1), range: RangeId(0) },
        ],
        terms: vec![
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![
                    Index { id: IndexId(2), range: RangeId(0) },
                    Index { id: IndexId(3), range: RangeId(0) },
                ],
                factors: vec![
                    Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(2)] },
                    Factor { tensor: TensorId(1), indices: vec![IndexId(2), IndexId(3)] },
                    Factor { tensor: TensorId(2), indices: vec![IndexId(3), IndexId(1)] },
                ],
            },
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![
                    Index { id: IndexId(4), range: RangeId(0) },
                    Index { id: IndexId(5), range: RangeId(0) },
                ],
                factors: vec![
                    Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(4)] },
                    Factor { tensor: TensorId(1), indices: vec![IndexId(4), IndexId(5)] },
                    Factor { tensor: TensorId(2), indices: vec![IndexId(5), IndexId(1)] },
                ],
            },
        ],
    };

    let tensors = vec![
        TensorInfo { id: TensorId(0), symmetry: vec![] },
        TensorInfo { id: TensorId(1), symmetry: vec![] },
        TensorInfo { id: TensorId(2), symmetry: vec![] },
    ];

    (def, tensors)
}

fn make_same_range_split() -> TermSplit {
    TermSplit {
        left_sub_term: Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index { id: IndexId(2), range: RangeId(0) }],
            factors: vec![Factor {
                tensor: TensorId(0),
                indices: vec![IndexId(0), IndexId(2)],
            }],
        },
        right_sub_term: Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![
                Index { id: IndexId(2), range: RangeId(0) },
                Index { id: IndexId(3), range: RangeId(0) },
            ],
            factors: vec![
                Factor { tensor: TensorId(1), indices: vec![IndexId(2), IndexId(3)] },
                Factor { tensor: TensorId(2), indices: vec![IndexId(3), IndexId(1)] },
            ],
        },
        last_step: LastStepIndices {
            left_ext: 1,
            right_ext: 2,
            sums: vec![RangeId(0)],
        },
    }
}

fn make_owner_stability_split_a() -> TermSplit {
    TermSplit {
        left_sub_term: Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![
                Index { id: IndexId(2), range: RangeId(0) },
                Index { id: IndexId(3), range: RangeId(1) },
            ],
            factors: vec![
                Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(2)] },
                Factor { tensor: TensorId(1), indices: vec![IndexId(2), IndexId(3)] },
            ],
        },
        right_sub_term: Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index { id: IndexId(3), range: RangeId(1) }],
            factors: vec![Factor {
                tensor: TensorId(2),
                indices: vec![IndexId(3), IndexId(1)],
            }],
        },
        last_step: LastStepIndices {
            left_ext: 1,
            right_ext: 2,
            sums: vec![RangeId(1)],
        },
    }
}

fn make_owner_stability_split_b() -> TermSplit {
    TermSplit {
        left_sub_term: Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![
                Index { id: IndexId(5), range: RangeId(0) },
                Index { id: IndexId(6), range: RangeId(1) },
            ],
            factors: vec![
                Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(5)] },
                Factor { tensor: TensorId(1), indices: vec![IndexId(5), IndexId(6)] },
            ],
        },
        right_sub_term: Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index { id: IndexId(6), range: RangeId(1) }],
            factors: vec![Factor {
                tensor: TensorId(2),
                indices: vec![IndexId(6), IndexId(1)],
            }],
        },
        last_step: LastStepIndices {
            left_ext: 1,
            right_ext: 2,
            sums: vec![RangeId(1)],
        },
    }
}

fn make_non_shared_sum_id_split() -> TermSplit {
    TermSplit {
        left_sub_term: Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index { id: IndexId(2), range: RangeId(0) }],
            factors: vec![Factor {
                tensor: TensorId(0),
                indices: vec![IndexId(0), IndexId(0)],
            }],
        },
        right_sub_term: Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index { id: IndexId(2), range: RangeId(0) }],
            factors: vec![Factor {
                tensor: TensorId(1),
                indices: vec![IndexId(1), IndexId(1)],
            }],
        },
        last_step: LastStepIndices {
            left_ext: 1,
            right_ext: 2,
            sums: vec![],
        },
    }
}

#[test]
fn test_canon_term_normalizes_dummy_names() {
    let (def, tensors) = make_context_def_and_tensors();
    let cx = build_canon_def_context(&def, &tensors);

    let left = canon_term(&def.terms[0], &cx);
    let right = canon_term(&def.terms[1], &cx);

    assert_eq!(left, right);
}

#[test]
fn test_canon_term_normalizes_sum_index_order() {
    let (mut def, tensors) = make_context_def_and_tensors();
    def.terms[1].sum_indices.reverse();

    let cx = build_canon_def_context(&def, &tensors);
    let left = canon_term(&def.terms[0], &cx);
    let right = canon_term(&def.terms[1], &cx);

    assert_eq!(left, right);
}

#[test]
fn test_canon_term_distinguishes_external_and_dummy_positions() {
    let def = TensorDef {
        base: TensorId(100),
        ext_indices: vec![Index { id: IndexId(0), range: RangeId(0) }],
        terms: vec![
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![Index { id: IndexId(1), range: RangeId(0) }],
                factors: vec![
                    Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(1)] },
                    Factor { tensor: TensorId(0), indices: vec![IndexId(1), IndexId(1)] },
                ],
            },
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![Index { id: IndexId(2), range: RangeId(0) }],
                factors: vec![
                    Factor { tensor: TensorId(0), indices: vec![IndexId(2), IndexId(2)] },
                    Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(2)] },
                ],
            },
        ],
    };
    let tensors = vec![TensorInfo { id: TensorId(0), symmetry: vec![] }];
    let cx = build_canon_def_context(&def, &tensors);

    let left = canon_term(&def.terms[0], &cx);
    let right = canon_term(&def.terms[1], &cx);

    assert_eq!(left, right);
}

#[test]
fn test_canon_term_is_deterministic_with_tied_factors_and_symmetry() {
    let def = TensorDef {
        base: TensorId(101),
        ext_indices: vec![
            Index { id: IndexId(0), range: RangeId(0) },
            Index { id: IndexId(1), range: RangeId(0) },
        ],
        terms: vec![
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![
                    Index { id: IndexId(2), range: RangeId(0) },
                    Index { id: IndexId(3), range: RangeId(0) },
                ],
                factors: vec![
                    Factor { tensor: TensorId(0), indices: vec![IndexId(2), IndexId(0)] },
                    Factor { tensor: TensorId(0), indices: vec![IndexId(3), IndexId(1)] },
                    Factor { tensor: TensorId(1), indices: vec![IndexId(2), IndexId(3)] },
                ],
            },
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![
                    Index { id: IndexId(5), range: RangeId(0) },
                    Index { id: IndexId(4), range: RangeId(0) },
                ],
                factors: vec![
                    Factor { tensor: TensorId(0), indices: vec![IndexId(4), IndexId(1)] },
                    Factor { tensor: TensorId(1), indices: vec![IndexId(4), IndexId(5)] },
                    Factor { tensor: TensorId(0), indices: vec![IndexId(5), IndexId(0)] },
                ],
            },
        ],
    };
    let tensors = vec![
        TensorInfo { id: TensorId(0), symmetry: vec![] },
        TensorInfo {
            id: TensorId(1),
            symmetry: vec![SymGenerator {
                perm: vec![1, 0],
                action: SymAction::Identity,
            }],
        },
    ];
    let cx = build_canon_def_context(&def, &tensors);

    let left = canon_term(&def.terms[0], &cx);
    let right = canon_term(&def.terms[1], &cx);

    assert_eq!(left, right);
}

#[test]
fn test_canon_split_left_assigned_uses_consistent_shared_name() {
    let (def, tensors) = make_context_def_and_tensors();
    let cx = build_canon_def_context(&def, &tensors);
    let split = make_split();

    let pair = canon_split(&split, &cx);

    assert_eq!(pair.left_assigned.last_step, split.last_step);

    let left_shared = pair.left_assigned.left_sub_term.factors[0].indices[1];
    let right_shared = pair.left_assigned.right_sub_term.factors[0].indices[0];
    let right_private = pair.left_assigned.right_sub_term.factors[0].indices[1];

    assert_eq!(left_shared, right_shared);
    assert!(pair.left_assigned.left_sub_term.sum_indices.is_empty());
    assert_eq!(pair.left_assigned.right_sub_term.sum_indices.len(), 1);
    assert_eq!(pair.left_assigned.right_sub_term.sum_indices[0].range, RangeId(1));
    assert_eq!(pair.left_assigned.right_sub_term.sum_indices[0].id, right_private);
    assert_ne!(right_private, left_shared);
    assert_eq!(pair.left_assigned.right_sub_term.factors[1].indices[0], right_private);
}

#[test]
fn test_canon_split_right_assigned_uses_consistent_shared_name() {
    let (def, tensors) = make_context_def_and_tensors();
    let cx = build_canon_def_context(&def, &tensors);
    let split = make_split();

    let pair = canon_split(&split, &cx);

    assert_eq!(pair.right_assigned.last_step, split.last_step);

    let left_shared = pair.right_assigned.left_sub_term.factors[0].indices[1];
    let right_shared = pair.right_assigned.right_sub_term.factors[0].indices[0];
    let right_private = pair.right_assigned.right_sub_term.factors[0].indices[1];

    assert_eq!(left_shared, right_shared);
    assert!(pair.right_assigned.left_sub_term.sum_indices.is_empty());
    assert_eq!(pair.right_assigned.right_sub_term.sum_indices.len(), 1);
    assert_eq!(pair.right_assigned.right_sub_term.sum_indices[0].range, RangeId(1));
    assert_eq!(pair.right_assigned.right_sub_term.sum_indices[0].id, right_private);
    assert_ne!(right_private, left_shared);
    assert_eq!(pair.right_assigned.right_sub_term.factors[1].indices[0], right_private);
}

#[test]
fn test_canon_split_right_side_private_dummy_avoids_same_range_shared_collision() {
    let (def, tensors) = make_same_range_context_def_and_tensors();
    let cx = build_canon_def_context(&def, &tensors);
    let split = make_same_range_split();

    let pair = canon_split(&split, &cx);

    let shared = pair.right_assigned.left_sub_term.factors[0].indices[1];
    let private = pair.right_assigned.right_sub_term.factors[0].indices[1];

    assert_eq!(pair.right_assigned.right_sub_term.sum_indices.len(), 1);
    assert_eq!(pair.right_assigned.right_sub_term.sum_indices[0].range, RangeId(0));
    assert_eq!(pair.right_assigned.right_sub_term.sum_indices[0].id, private);
    assert_eq!(pair.right_assigned.right_sub_term.factors[1].indices[0], private);
    assert_ne!(shared, private);
}

#[test]
fn test_canon_split_owner_side_is_stable_across_followers() {
    let (def, tensors) = make_context_def_and_tensors();
    let cx = build_canon_def_context(&def, &tensors);
    let split_a = make_owner_stability_split_a();
    let split_b = make_owner_stability_split_b();

    let pair_a = canon_split(&split_a, &cx);
    let pair_b = canon_split(&split_b, &cx);

    assert_eq!(
        pair_a.left_assigned.left_sub_term,
        pair_b.left_assigned.left_sub_term
    );
}

#[test]
fn test_canon_split_ignores_sum_only_overlap_when_deriving_shared_interface() {
    let (def, tensors) = make_same_range_context_def_and_tensors();
    let cx = build_canon_def_context(&def, &tensors);
    let split = make_non_shared_sum_id_split();

    let pair = canon_split(&split, &cx);

    assert_eq!(pair.left_assigned.left_sub_term.sum_indices.len(), 1);
    assert_eq!(pair.left_assigned.right_sub_term.sum_indices.len(), 1);
}

#[test]
fn test_canonical_term_key_orders_terms_lexicographically() {
    let t1 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![],
        factors: vec![Factor { tensor: TensorId(0), indices: vec![IndexId(0)] }],
    };
    let t2 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![],
        factors: vec![Factor { tensor: TensorId(1), indices: vec![IndexId(0)] }],
    };

    assert!(canonical_term_key(&t1) < canonical_term_key(&t2));
}
