use num::rational::Ratio;

use rustymill::biclique_action::{
    apply_action_selection, next_action_decision, rewrite_from_action_selection,
    validate_action_selection, ActionDecision, FactorizationRewrite, StructuredAction,
};
use rustymill::repr::*;

fn factor(tensor: u32, indices: &[u32]) -> Factor {
    Factor {
        tensor: TensorId(tensor),
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn index(id: u32, range: RangeId) -> Index {
    Index {
        id: IndexId(id),
        range,
    }
}

fn term(coeff_num: i64, coeff_den: i64, sum_indices: &[Index], factors: Vec<Factor>) -> Term {
    Term {
        coeff: Ratio::new(coeff_num, coeff_den),
        sum_indices: sum_indices.to_vec(),
        factors,
    }
}

fn make_non_actionable_comp() -> TensorComputation {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let target = comp.add_tensor(vec![]);
    let simple = comp.add_tensor(vec![]);

    comp.add_definition(
        target,
        vec![index(0, occ), index(1, occ)],
        vec![term(1, 1, &[], vec![factor(simple.0, &[0, 1])])],
    );
    comp
}

fn make_actionable_comp() -> TensorComputation {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let target = comp.add_tensor(vec![]);
    let x = comp.add_tensor(vec![]);
    let y = comp.add_tensor(vec![]);
    let p = comp.add_tensor(vec![]);
    let q = comp.add_tensor(vec![]);
    let remainder = comp.add_tensor(vec![]);

    let ext = vec![index(0, occ), index(1, occ)];
    let shared = [index(2, occ)];

    let make_biclique_term = |left: TensorId, right: TensorId| {
        term(
            1,
            1,
            &shared,
            vec![
                Factor {
                    tensor: left,
                    indices: vec![IndexId(0), IndexId(2)],
                },
                Factor {
                    tensor: right,
                    indices: vec![IndexId(2), IndexId(1)],
                },
            ],
        )
    };

    comp.add_definition(
        target,
        ext,
        vec![
            make_biclique_term(x, p),
            make_biclique_term(y, p),
            make_biclique_term(x, q),
            make_biclique_term(y, q),
            term(
                1,
                1,
                &[],
                vec![Factor {
                    tensor: remainder,
                    indices: vec![IndexId(0), IndexId(1)],
                }],
            ),
        ],
    );
    comp
}

fn make_reordered_biclique_term_comp() -> TensorComputation {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let target = comp.add_tensor(vec![]);
    let x = comp.add_tensor(vec![]);
    let y = comp.add_tensor(vec![]);
    let p = comp.add_tensor(vec![]);
    let q = comp.add_tensor(vec![]);
    let remainder = comp.add_tensor(vec![]);

    let ext = vec![index(0, occ), index(1, occ)];
    let shared = [index(2, occ)];

    let make_biclique_term = |left: TensorId, right: TensorId| {
        term(
            1,
            1,
            &shared,
            vec![
                Factor {
                    tensor: left,
                    indices: vec![IndexId(0), IndexId(2)],
                },
                Factor {
                    tensor: right,
                    indices: vec![IndexId(2), IndexId(1)],
                },
            ],
        )
    };

    comp.add_definition(
        target,
        ext,
        vec![
            make_biclique_term(x, q),
            make_biclique_term(y, q),
            make_biclique_term(x, p),
            make_biclique_term(y, p),
            term(
                1,
                1,
                &[],
                vec![Factor {
                    tensor: remainder,
                    indices: vec![IndexId(0), IndexId(1)],
                }],
            ),
        ],
    );
    comp
}

fn make_alpha_equivalent_dummy_comp() -> TensorComputation {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let target = comp.add_tensor(vec![]);
    let x = comp.add_tensor(vec![]);
    let y = comp.add_tensor(vec![]);
    let p = comp.add_tensor(vec![]);
    let q = comp.add_tensor(vec![]);

    let ext = vec![index(0, occ), index(1, occ)];
    let c = [index(2, occ)];
    let d = [index(3, occ)];

    comp.add_definition(
        target,
        ext,
        vec![
            term(
                1,
                1,
                &c,
                vec![
                    Factor {
                        tensor: x,
                        indices: vec![IndexId(0), IndexId(2)],
                    },
                    Factor {
                        tensor: p,
                        indices: vec![IndexId(2), IndexId(1)],
                    },
                ],
            ),
            term(
                1,
                1,
                &c,
                vec![
                    Factor {
                        tensor: y,
                        indices: vec![IndexId(0), IndexId(2)],
                    },
                    Factor {
                        tensor: p,
                        indices: vec![IndexId(2), IndexId(1)],
                    },
                ],
            ),
            term(
                1,
                1,
                &d,
                vec![
                    Factor {
                        tensor: x,
                        indices: vec![IndexId(0), IndexId(3)],
                    },
                    Factor {
                        tensor: q,
                        indices: vec![IndexId(3), IndexId(1)],
                    },
                ],
            ),
            term(
                1,
                1,
                &d,
                vec![
                    Factor {
                        tensor: y,
                        indices: vec![IndexId(0), IndexId(3)],
                    },
                    Factor {
                        tensor: q,
                        indices: vec![IndexId(3), IndexId(1)],
                    },
                ],
            ),
        ],
    );
    comp
}

#[test]
fn test_crate_surface_exposes_biclique_action_api() {
    let next_fn: fn(&TensorComputation, usize) -> Option<ActionDecision> = next_action_decision;
    let validate_fn: fn(&ActionDecision, &StructuredAction) -> Result<(), String> =
        validate_action_selection;
    let rewrite_fn: fn(
        &TensorComputation,
        usize,
        &ActionDecision,
        &StructuredAction,
    ) -> Result<FactorizationRewrite, String> = rewrite_from_action_selection;
    let apply_fn: fn(
        &mut TensorComputation,
        usize,
        &ActionDecision,
        &StructuredAction,
    ) -> Result<(), String> = apply_action_selection;

    let action = StructuredAction {
        biclique_index: 0,
        left_mask: vec![true],
        right_mask: vec![true],
    };

    let _ = (next_fn, validate_fn, rewrite_fn, apply_fn, action);
}

#[test]
fn test_next_action_decision_returns_none_when_no_biclique_exists() {
    let comp = make_non_actionable_comp();

    assert!(next_action_decision(&comp, 0).is_none());
}

#[test]
fn test_next_action_decision_exports_one_faithful_template_per_maximal_biclique() {
    let comp = make_actionable_comp();
    let decision = next_action_decision(&comp, 0).expect("expected an action decision");

    assert_eq!(decision.def_index(), 0);
    assert_eq!(decision.candidate_bicliques().len(), 2);

    for template in decision.candidate_bicliques() {
        assert_eq!(template.definitions().len(), 3);

        let left = &template.definitions()[0];
        let right = &template.definitions()[1];
        let rewritten = &template.definitions()[2];

        assert_eq!(left.terms.len(), 2);
        assert_eq!(right.terms.len(), 2);
        assert_eq!(rewritten.terms.len(), 2);
        assert_eq!(rewritten.terms[1].factors.len(), 1);
    }
}

#[test]
fn test_next_action_decision_keeps_distinct_maximal_bicliques_even_when_templates_match() {
    let comp = make_actionable_comp();
    let decision = next_action_decision(&comp, 0).expect("expected an action decision");

    assert_eq!(decision.candidate_bicliques().len(), 2);
    assert_eq!(
        decision.candidate_bicliques()[0],
        decision.candidate_bicliques()[1]
    );
}

#[test]
fn test_next_action_decision_preserves_remainder_term_in_definition_two() {
    let comp = make_actionable_comp();
    let decision = next_action_decision(&comp, 0).expect("expected an action decision");
    let rewritten = &decision.candidate_bicliques()[0].definitions()[2];

    assert_eq!(rewritten.terms.len(), 2);
    assert_eq!(rewritten.terms[1].coeff, Ratio::from_integer(1));
    assert_eq!(rewritten.terms[1].sum_indices.len(), 0);
    assert_eq!(rewritten.terms[1].factors.len(), 1);
}

#[test]
fn test_next_action_decision_is_deterministic_for_repeated_calls() {
    let comp = make_actionable_comp();

    let first = next_action_decision(&comp, 0).expect("first decision missing");
    let second = next_action_decision(&comp, 0).expect("second decision missing");

    assert_eq!(first.def_index(), second.def_index());
    assert_eq!(first.candidate_bicliques().len(), 2);
    assert_eq!(first.candidate_bicliques(), second.candidate_bicliques());
}

#[test]
fn test_next_action_decision_accepts_alpha_equivalent_dummy_ids() {
    let comp = make_alpha_equivalent_dummy_comp();
    let decision = next_action_decision(&comp, 0).expect("expected an action decision");

    assert_eq!(decision.def_index(), 0);
    assert_eq!(decision.candidate_bicliques().len(), 2);
    assert_eq!(
        decision.candidate_bicliques()[0],
        decision.candidate_bicliques()[1]
    );
    assert_eq!(decision.candidate_bicliques()[0].definitions().len(), 3);
}

#[test]
fn test_validate_action_selection_rejects_out_of_range_and_empty_masks() {
    let comp = make_actionable_comp();
    let decision = next_action_decision(&comp, 0).expect("expected an action decision");

    let out_of_range = StructuredAction {
        biclique_index: 7,
        left_mask: vec![true, true],
        right_mask: vec![true, true],
    };
    let empty_left = StructuredAction {
        biclique_index: 0,
        left_mask: vec![false, false],
        right_mask: vec![true, true],
    };
    let empty_right = StructuredAction {
        biclique_index: 0,
        left_mask: vec![true, true],
        right_mask: vec![false, false],
    };

    assert!(validate_action_selection(&decision, &out_of_range).is_err());
    assert!(validate_action_selection(&decision, &empty_left).is_err());
    assert!(validate_action_selection(&decision, &empty_right).is_err());
}

#[test]
fn test_validate_action_selection_rejects_mask_length_mismatch() {
    let comp = make_actionable_comp();
    let decision = next_action_decision(&comp, 0).expect("expected an action decision");

    let bad = StructuredAction {
        biclique_index: 0,
        left_mask: vec![true],
        right_mask: vec![true, true, false],
    };

    assert!(validate_action_selection(&decision, &bad).is_err());
}

#[test]
fn test_rewrite_from_action_selection_consumes_exact_selected_rectangle() {
    let comp = make_actionable_comp();
    let decision = next_action_decision(&comp, 0).expect("expected an action decision");
    let action = StructuredAction {
        biclique_index: 0,
        left_mask: vec![true, false],
        right_mask: vec![true, true],
    };

    let rewrite =
        rewrite_from_action_selection(&comp, 0, &decision, &action).expect("rewrite missing");

    assert_eq!(rewrite.def_index, 0);
    assert_eq!(rewrite.consumed_term_indices, vec![0, 2]);
    assert_eq!(rewrite.new_tensor_count, 2);
    assert_eq!(rewrite.replacement_definitions.len(), 3);
    assert_eq!(rewrite.replacement_definitions[0].terms.len(), 1);
    assert_eq!(rewrite.replacement_definitions[1].terms.len(), 2);
    assert_eq!(rewrite.replacement_definitions[2].terms.len(), 4);
}

#[test]
fn test_rewrite_from_action_selection_rejects_reordered_stale_decision() {
    let original = make_actionable_comp();
    let reordered = make_reordered_biclique_term_comp();

    let original_decision =
        next_action_decision(&original, 0).expect("expected original action decision");

    let action = StructuredAction {
        biclique_index: 0,
        left_mask: vec![true, false],
        right_mask: vec![true, true],
    };

    let err = rewrite_from_action_selection(&reordered, 0, &original_decision, &action)
        .expect_err("reordered stale decision should be rejected");
    assert!(err.contains("candidate") || err.contains("stale"));
}

#[test]
fn test_apply_action_selection_rewrites_definition_and_inserts_intermediates() {
    let mut comp = make_actionable_comp();
    let decision = next_action_decision(&comp, 0).expect("expected an action decision");
    let action = StructuredAction {
        biclique_index: 0,
        left_mask: vec![true, true],
        right_mask: vec![true, true],
    };

    apply_action_selection(&mut comp, 0, &decision, &action).expect("apply should succeed");

    assert_eq!(comp.definitions().len(), 3);
    assert_eq!(comp.definitions()[0].terms.len(), 2);
    assert_eq!(comp.definitions()[1].terms.len(), 2);
    assert_eq!(comp.definitions()[2].terms.len(), 2);
}

#[test]
fn test_apply_action_selection_simplifies_only_explicit_safe_single_term_side() {
    let mut comp = make_actionable_comp();
    let initial_next_tensor = comp.next_tensor_id();
    let decision = next_action_decision(&comp, 0).expect("expected an action decision");
    let action = StructuredAction {
        biclique_index: 0,
        left_mask: vec![true, false],
        right_mask: vec![true, true],
    };

    apply_action_selection(&mut comp, 0, &decision, &action).expect("apply should succeed");

    assert_eq!(comp.definitions().len(), 2);
    assert_eq!(comp.definitions()[0].terms.len(), 2);
    assert_eq!(comp.definitions()[1].terms.len(), 4);
    assert_eq!(comp.next_tensor_id().0, initial_next_tensor.0 + 1);
    assert_eq!(comp.definitions()[0].base, initial_next_tensor);
}

#[test]
fn test_apply_action_selection_rejects_stale_decision_after_state_change() {
    let mut comp = make_actionable_comp();
    let decision = next_action_decision(&comp, 0).expect("expected an action decision");
    let action = StructuredAction {
        biclique_index: 0,
        left_mask: vec![true, true],
        right_mask: vec![true, true],
    };

    apply_action_selection(&mut comp, 0, &decision, &action).expect("first apply should succeed");

    let stale = apply_action_selection(&mut comp, 0, &decision, &action);
    assert!(stale.is_err());
}
