use num::rational::Ratio;
use rustymill::repr::*;
use rustymill::rl_parenth::{enumerate_splits, LastStepIndices};

fn make_abc_term() -> (TensorDef, Term) {
    let occ = RangeId(0);
    let virt = RangeId(1);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);
    let d = IndexId(3);

    let def = TensorDef {
        base: TensorId(99),
        ext_indices: vec![
            Index { id: a, range: occ },
            Index { id: b, range: virt },
        ],
        terms: Vec::new(),
    };

    let term = Term {
        coeff: Ratio::new(7, 3),
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

    (def, term)
}

#[test]
fn test_enumerate_splits_three_factors_returns_three_unique_splits() {
    let (def, term) = make_abc_term();

    let splits = enumerate_splits(&term, &def);

    assert_eq!(splits.len(), 3);
    assert_eq!(
        splits.iter().map(|s| &s.last_step).collect::<Vec<_>>(),
        vec![
            &LastStepIndices { left_ext: 0b00, right_ext: 0b11, sums: vec![RangeId(0), RangeId(1)] },
            &LastStepIndices { left_ext: 0b01, right_ext: 0b10, sums: vec![RangeId(0)] },
            &LastStepIndices { left_ext: 0b01, right_ext: 0b10, sums: vec![RangeId(1)] },
        ]
    );
}

#[test]
fn test_enumerate_splits_preserves_factor_order_and_resets_coeff() {
    let (def, term) = make_abc_term();

    let splits = enumerate_splits(&term, &def);
    let split = splits
        .iter()
        .find(|s| s.last_step == LastStepIndices { left_ext: 0b01, right_ext: 0b10, sums: vec![RangeId(0)] })
        .unwrap();

    assert_eq!(split.left_sub_term.coeff, Ratio::from_integer(1));
    assert_eq!(split.right_sub_term.coeff, Ratio::from_integer(1));

    assert_eq!(
        split.left_sub_term.factors,
        vec![Factor { tensor: TensorId(0), indices: vec![IndexId(0), IndexId(2)] }]
    );
    assert_eq!(
        split.right_sub_term.factors,
        vec![
            Factor { tensor: TensorId(1), indices: vec![IndexId(2), IndexId(3)] },
            Factor { tensor: TensorId(2), indices: vec![IndexId(3), IndexId(1)] },
        ]
    );
}

#[test]
fn test_enumerate_splits_filters_sum_indices_to_selected_factors() {
    let (def, term) = make_abc_term();

    let splits = enumerate_splits(&term, &def);
    let split = splits
        .iter()
        .find(|s| s.last_step == LastStepIndices { left_ext: 0b01, right_ext: 0b10, sums: vec![RangeId(0)] })
        .unwrap();

    assert_eq!(
        split.left_sub_term.sum_indices,
        vec![Index { id: IndexId(2), range: RangeId(0) }]
    );
    assert_eq!(
        split.right_sub_term.sum_indices,
        vec![Index { id: IndexId(2), range: RangeId(0) }, Index { id: IndexId(3), range: RangeId(1) }]
    );
}
