use rustymill::repr::*;
use rustymill::optimize::*;
use rustymill::cost::total_cost;
use num::rational::Ratio;

#[test]
fn test_greedy_optimize_shared_factor() {
    // t[a,b] = X[a,c]*Z[c,b] + Y[a,c]*Z[c,b]
    // Should factor out Z: tau[a,c] = X[a,c] + Y[a,c], t[a,b] = tau[a,c]*Z[c,b]
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _x = comp.add_tensor(&[occ, occ], vec![]);
    let _y = comp.add_tensor(&[occ, occ], vec![]);
    let _z = comp.add_tensor(&[occ, occ], vec![]);
    let t = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);

    let ext = vec![
        Index { id: a, range: occ },
        Index { id: b, range: occ },
    ];

    comp.add_definition(
        t,
        ext,
        vec![
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![Index { id: c, range: occ }],
                factors: vec![
                    Factor { tensor: TensorId(0), indices: vec![a, c] },
                    Factor { tensor: TensorId(2), indices: vec![c, b] },
                ],
            },
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![Index { id: c, range: occ }],
                factors: vec![
                    Factor { tensor: TensorId(1), indices: vec![a, c] },
                    Factor { tensor: TensorId(2), indices: vec![c, b] },
                ],
            },
        ],
    );

    let cost_before = total_cost(&comp);
    let n = greedy_optimize(&mut comp);

    assert!(n >= 1, "Should apply at least 1 factorization");
    let cost_after = total_cost(&comp);
    assert!(cost_after < cost_before, "Cost should decrease");
}

#[test]
fn test_greedy_optimize_full_biclique() {
    // t[a,b] = 4*X*U + 2*X*V - 2*Y*U - Y*V
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _x = comp.add_tensor(&[occ, occ], vec![]);
    let _y = comp.add_tensor(&[occ, occ], vec![]);
    let _u = comp.add_tensor(&[occ, occ], vec![]);
    let _v = comp.add_tensor(&[occ, occ], vec![]);
    let t = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);

    let ext = vec![
        Index { id: a, range: occ },
        Index { id: b, range: occ },
    ];

    let make_term = |coeff: i64, left: TensorId, right: TensorId| -> Term {
        Term {
            coeff: Ratio::from_integer(coeff),
            sum_indices: vec![Index { id: c, range: occ }],
            factors: vec![
                Factor { tensor: left, indices: vec![a, c] },
                Factor { tensor: right, indices: vec![c, b] },
            ],
        }
    };

    comp.add_definition(
        t,
        ext,
        vec![
            make_term(4, TensorId(0), TensorId(2)),
            make_term(2, TensorId(0), TensorId(3)),
            make_term(-2, TensorId(1), TensorId(2)),
            make_term(-1, TensorId(1), TensorId(3)),
        ],
    );

    let cost_before = total_cost(&comp);
    let n = greedy_optimize(&mut comp);

    assert!(n >= 1);
    let cost_after = total_cost(&comp);
    assert!(cost_after < cost_before);

    // Should have created intermediates
    assert!(comp.definitions().len() > 1);
}

#[test]
fn test_greedy_optimize_nothing_to_do() {
    // Single term, nothing to factorize
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
            factors: vec![Factor {
                tensor: TensorId(0),
                indices: vec![a, b],
            }],
        }],
    );

    let n = greedy_optimize(&mut comp);
    assert_eq!(n, 0);
    assert_eq!(comp.definitions().len(), 1);
}
