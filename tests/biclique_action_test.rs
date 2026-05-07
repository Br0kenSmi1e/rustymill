use num::rational::Ratio;

use rustymill::biclique_action::{
    apply_factorization_rewrite, build_rewrite_from_decision, next_action_space,
    validate_decision, ActionSpace, Decision, Factorization, FactorizationRewrite,
};
use rustymill::repr::{
    Factor, Index, IndexId, TensorComputation, TensorDef, TensorId, Term,
};

fn make_non_actionable_comp() -> TensorComputation {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let target = comp.add_tensor(vec![]);
    let simple = comp.add_tensor(vec![]);

    comp.add_definition(
        target,
        vec![
            Index {
                id: IndexId(0),
                range: occ,
            },
            Index {
                id: IndexId(1),
                range: occ,
            },
        ],
        vec![Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![],
            factors: vec![Factor {
                tensor: simple,
                indices: vec![IndexId(0), IndexId(1)],
            }],
        }],
    );

    comp
}

fn make_actionable_comp_with_term_order(term_order: [usize; 4]) -> TensorComputation {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let target = comp.add_tensor(vec![]);
    let x = comp.add_tensor(vec![]);
    let y = comp.add_tensor(vec![]);
    let p = comp.add_tensor(vec![]);
    let q = comp.add_tensor(vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);

    let ext_indices = vec![
        Index {
            id: a,
            range: occ,
        },
        Index {
            id: b,
            range: occ,
        },
    ];

    let template_terms = vec![
        Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index {
                id: c,
                range: occ,
            }],
            factors: vec![
                Factor {
                    tensor: x,
                    indices: vec![a, c],
                },
                Factor {
                    tensor: p,
                    indices: vec![c, b],
                },
            ],
        },
        Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index {
                id: c,
                range: occ,
            }],
            factors: vec![
                Factor {
                    tensor: y,
                    indices: vec![a, c],
                },
                Factor {
                    tensor: p,
                    indices: vec![c, b],
                },
            ],
        },
        Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index {
                id: c,
                range: occ,
            }],
            factors: vec![
                Factor {
                    tensor: x,
                    indices: vec![a, c],
                },
                Factor {
                    tensor: q,
                    indices: vec![c, b],
                },
            ],
        },
        Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![Index {
                id: c,
                range: occ,
            }],
            factors: vec![
                Factor {
                    tensor: y,
                    indices: vec![a, c],
                },
                Factor {
                    tensor: q,
                    indices: vec![c, b],
                },
            ],
        },
    ];

    let terms = term_order
        .into_iter()
        .map(|idx| template_terms[idx].clone())
        .collect();

    comp.add_definition(target, ext_indices, terms);
    comp
}

fn make_actionable_comp() -> TensorComputation {
    make_actionable_comp_with_term_order([0, 1, 2, 3])
}

fn actionable_space() -> ActionSpace {
    let comp = make_actionable_comp();
    next_action_space(&comp, 0).expect("expected actionable biclique space for fixture")
}

#[test]
fn test_crate_surface_exposes_action_space_api() {
    let next_fn: fn(&TensorComputation, usize) -> Option<ActionSpace> = next_action_space;
    let validate_fn: fn(&ActionSpace, &Decision) -> Result<(), String> = validate_decision;
    let rewrite_fn: fn(
        &TensorComputation,
        &ActionSpace,
        &Decision,
    ) -> Result<FactorizationRewrite, String> = build_rewrite_from_decision;
    let apply_fn: fn(&mut TensorComputation, FactorizationRewrite) -> Result<(), String> =
        apply_factorization_rewrite;

    let template = Factorization {
        left_definition: TensorDef {
            base: TensorId(0),
            ext_indices: vec![],
            terms: vec![],
        },
        right_definition: TensorDef {
            base: TensorId(1),
            ext_indices: vec![],
            terms: vec![],
        },
        rewritten_definition: TensorDef {
            base: TensorId(2),
            ext_indices: vec![],
            terms: vec![],
        },
    };
    let decision = Decision {
        candidate_index: 0,
        left_mask: vec![true],
        right_mask: vec![true],
    };

    let _ = (next_fn, validate_fn, rewrite_fn, apply_fn, template, decision);
}

#[test]
fn test_validate_decision_rejects_out_of_range_candidate_index() {
    let space = actionable_space();
    let decision = Decision {
        candidate_index: space.candidate_templates.len(),
        left_mask: vec![true; space.candidate_templates[0].left_definition.terms.len()],
        right_mask: vec![true; space.candidate_templates[0].right_definition.terms.len()],
    };

    let err = validate_decision(&space, &decision).expect_err("expected candidate_index check");
    assert!(err.contains("candidate_index"), "{err}");
    assert!(err.contains("out of range"), "{err}");
}

#[test]
fn test_validate_decision_rejects_mask_length_mismatch() {
    let space = actionable_space();
    let template = &space.candidate_templates[0];
    let decision = Decision {
        candidate_index: 0,
        left_mask: vec![true; template.left_definition.terms.len().saturating_sub(1)],
        right_mask: vec![true; template.right_definition.terms.len()],
    };

    let err = validate_decision(&space, &decision).expect_err("expected mask length check");
    assert!(err.contains("mask"), "{err}");
    assert!(err.contains("length"), "{err}");
}

#[test]
fn test_validate_decision_rejects_empty_side() {
    let space = actionable_space();
    let template = &space.candidate_templates[0];
    let decision = Decision {
        candidate_index: 0,
        left_mask: vec![false; template.left_definition.terms.len()],
        right_mask: vec![true; template.right_definition.terms.len()],
    };

    let err = validate_decision(&space, &decision).expect_err("expected empty side check");
    assert!(err.contains("empty"), "{err}");
    assert!(err.contains("side"), "{err}");
}

#[test]
fn test_next_action_space_returns_none_when_no_biclique_exists() {
    let comp = make_non_actionable_comp();

    assert!(next_action_space(&comp, 0).is_none());
}

#[test]
fn test_next_action_space_exports_factorization_templates() {
    let comp = make_actionable_comp();

    let space = next_action_space(&comp, 0)
        .expect("expected actionable biclique space for full 2x2 factorization fixture");

    assert_eq!(space.def_index, 0);
    assert!(!space.candidate_templates.is_empty());

    let template = &space.candidate_templates[0];
    assert!(!template.left_definition.terms.is_empty());
    assert!(!template.right_definition.terms.is_empty());
    assert_eq!(template.rewritten_definition.base, comp.definitions()[0].base);
}

#[test]
fn test_build_rewrite_from_decision_full_biclique_matches_template() {
    let comp = make_actionable_comp();
    let space =
        next_action_space(&comp, 0).expect("expected actionable biclique space for fixture");
    let template = space.candidate_templates[0].clone();
    let decision = Decision {
        candidate_index: 0,
        left_mask: vec![true; template.left_definition.terms.len()],
        right_mask: vec![true; template.right_definition.terms.len()],
    };

    let rewrite = build_rewrite_from_decision(&comp, &space, &decision)
        .expect("expected rewrite for full-biclique selection");

    assert_eq!(rewrite.def_index, 0);
    assert_eq!(rewrite.factorization, template);
}

#[test]
fn test_build_rewrite_from_decision_strict_left_subset_shrinks_left_side() {
    let comp = make_actionable_comp();
    let space =
        next_action_space(&comp, 0).expect("expected actionable biclique space for fixture");
    let template = space.candidate_templates[0].clone();
    assert!(
        template.left_definition.terms.len() >= 2,
        "fixture must expose a strict left-side subset",
    );

    let mut left_mask = vec![false; template.left_definition.terms.len()];
    left_mask[0] = true;
    let decision = Decision {
        candidate_index: 0,
        left_mask,
        right_mask: vec![true; template.right_definition.terms.len()],
    };

    let rewrite = build_rewrite_from_decision(&comp, &space, &decision)
        .expect("expected rewrite for strict left-side subset");

    assert_eq!(rewrite.factorization.left_definition.terms.len(), 1);
    assert_eq!(
        rewrite.factorization.left_definition.terms,
        vec![template.left_definition.terms[0].clone()]
    );
    assert_eq!(
        rewrite.factorization.right_definition.terms.len(),
        template.right_definition.terms.len()
    );
    assert_eq!(
        rewrite.factorization.right_definition,
        template.right_definition
    );
    assert!(rewrite.factorization.rewritten_definition.terms.len() > 1);
}

#[test]
fn test_apply_factorization_rewrite_installs_full_biclique_factorization() {
    let mut comp = make_actionable_comp();
    let original_definition_count = comp.definitions().len();
    let original_tensor_count = comp.tensors().len();

    let space =
        next_action_space(&comp, 0).expect("expected actionable biclique space for fixture");
    let template = &space.candidate_templates[0];
    let decision = Decision {
        candidate_index: 0,
        left_mask: vec![true; template.left_definition.terms.len()],
        right_mask: vec![true; template.right_definition.terms.len()],
    };

    let rewrite = build_rewrite_from_decision(&comp, &space, &decision)
        .expect("expected rewrite for full-biclique selection");
    let expected = rewrite.factorization.clone();

    apply_factorization_rewrite(&mut comp, rewrite)
        .expect("expected apply_factorization_rewrite to apply rewrite");

    assert_eq!(comp.tensors().len(), original_tensor_count + 2);
    assert_eq!(comp.definitions().len(), original_definition_count + 2);
    assert_eq!(comp.definitions()[0], expected.left_definition);
    assert_eq!(comp.definitions()[1], expected.right_definition);
    assert_eq!(comp.definitions()[2], expected.rewritten_definition);
}

#[test]
fn test_apply_factorization_rewrite_installs_strict_left_subset_factorization() {
    let mut comp = make_actionable_comp();
    let space =
        next_action_space(&comp, 0).expect("expected actionable biclique space for fixture");
    let template = &space.candidate_templates[0];
    assert!(
        template.left_definition.terms.len() >= 2,
        "fixture must expose a strict left-side subset",
    );

    let mut left_mask = vec![false; template.left_definition.terms.len()];
    left_mask[0] = true;
    let decision = Decision {
        candidate_index: 0,
        left_mask,
        right_mask: vec![true; template.right_definition.terms.len()],
    };

    let rewrite = build_rewrite_from_decision(&comp, &space, &decision)
        .expect("expected rewrite for strict left-side subset");
    let expected = rewrite.factorization.clone();

    apply_factorization_rewrite(&mut comp, rewrite)
        .expect("expected apply_factorization_rewrite to apply strict subset rewrite");

    assert_eq!(comp.definitions()[0], expected.left_definition);
    assert_eq!(comp.definitions()[1], expected.right_definition);
    assert_eq!(comp.definitions()[2], expected.rewritten_definition);
    assert_eq!(comp.definitions()[0].terms.len(), 1);
}

#[test]
fn test_apply_factorization_rewrite_allows_target_definition_drift() {
    let mut comp = make_actionable_comp();
    let space =
        next_action_space(&comp, 0).expect("expected actionable biclique space for fixture");
    let template = &space.candidate_templates[0];
    let decision = Decision {
        candidate_index: 0,
        left_mask: vec![true; template.left_definition.terms.len()],
        right_mask: vec![true; template.right_definition.terms.len()],
    };

    let rewrite = build_rewrite_from_decision(&comp, &space, &decision)
        .expect("expected rewrite for full-biclique selection");
    let expected = rewrite.factorization.clone();

    comp.definitions_mut()[0] = TensorDef {
        base: TensorId(1),
        ext_indices: vec![],
        terms: vec![Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![],
            factors: vec![],
        }],
    };

    apply_factorization_rewrite(&mut comp, rewrite)
        .expect("apply should trust the provided rewrite once boundary checks pass");

    assert_eq!(comp.definitions()[0], expected.left_definition);
    assert_eq!(comp.definitions()[1], expected.right_definition);
    assert_eq!(comp.definitions()[2], expected.rewritten_definition);
}

#[test]
fn test_apply_factorization_rewrite_rejects_out_of_range_def_index() {
    let mut comp = make_actionable_comp();
    let mut rewrite = FactorizationRewrite {
        def_index: comp.definitions().len(),
        factorization: Factorization {
            left_definition: TensorDef {
                base: comp.next_tensor_id(),
                ext_indices: vec![],
                terms: vec![],
            },
            right_definition: TensorDef {
                base: TensorId(comp.next_tensor_id().0 + 1),
                ext_indices: vec![],
                terms: vec![],
            },
            rewritten_definition: TensorDef {
                base: comp.definitions()[0].base,
                ext_indices: comp.definitions()[0].ext_indices.clone(),
                terms: comp.definitions()[0].terms.clone(),
            },
        },
    };

    let err = apply_factorization_rewrite(&mut comp, rewrite.clone())
        .expect_err("apply should reject rewrites with def_index out of range");
    assert!(err.contains("def_index"), "{err}");
    assert!(err.contains("out of range"), "{err}");

    rewrite.def_index = 0;
    rewrite.factorization.left_definition.base = TensorId(0);
    rewrite.factorization.right_definition.base = TensorId(1);

    let err = apply_factorization_rewrite(&mut comp, rewrite)
        .expect_err("fresh tensor ids should still be enforced");
    assert!(err.contains("tensor ids mismatch"), "{err}");
}
