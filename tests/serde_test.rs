use rustymill::repr::*;
use num::rational::Ratio;

fn build_sample_computation() -> TensorComputation {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let virt = comp.add_range(100);

    let sym = SymGenerator { perm: vec![1, 0], action: SymAction::Negate };
    let v = comp.add_tensor(vec![sym]);
    let t = comp.add_tensor(vec![]);
    let r = comp.add_tensor(vec![]);

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
