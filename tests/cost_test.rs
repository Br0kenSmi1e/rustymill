use rustymill::repr::*;
use rustymill::cost::{def_cost, total_cost};
use num::rational::Ratio;

fn simple_contraction() -> TensorComputation {
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
        vec![Index { id: a, range: occ }, Index { id: b, range: occ }],
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
    let cost = def_cost(&comp.definitions()[0], comp.ranges());
    assert_eq!(cost, 2100);
}

#[test]
fn test_def_cost_no_summation() {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _a = comp.add_tensor(&[occ, occ], vec![]);
    let t = comp.add_tensor(&[occ, occ], vec![]);
    let a = IndexId(0);
    let b = IndexId(1);
    comp.add_definition(
        t,
        vec![Index { id: a, range: occ }, Index { id: b, range: occ }],
        vec![Term {
            coeff: Ratio::from_integer(1),
            sum_indices: vec![],
            factors: vec![Factor { tensor: TensorId(0), indices: vec![a, b] }],
        }],
    );
    let cost = def_cost(&comp.definitions()[0], comp.ranges());
    assert_eq!(cost, 200);
}

#[test]
fn test_def_cost_two_terms_different_sums() {
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
        vec![Index { id: a, range: occ }, Index { id: b, range: occ }],
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
    comp.add_definition(t1, vec![Index { id: a, range: occ }, Index { id: b, range: occ }], terms.clone());
    comp.add_definition(t2, vec![Index { id: a, range: occ }, Index { id: b, range: occ }], terms);
    let cost = total_cost(&comp);
    assert_eq!(cost, 4200);
}

#[test]
fn test_def_cost_scalar_output() {
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
            sum_indices: vec![Index { id: a, range: occ }, Index { id: b, range: occ }],
            factors: vec![
                Factor { tensor: TensorId(0), indices: vec![a, b] },
                Factor { tensor: TensorId(1), indices: vec![a, b] },
            ],
        }],
    );
    let cost = def_cost(&comp.definitions()[0], comp.ranges());
    assert_eq!(cost, 201);
}
