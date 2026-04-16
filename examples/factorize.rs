use std::path::Path;

use rustymill::repr::*;
use rustymill::optimize::greedy_optimize;
use rustymill::cost::total_cost;
use rustymill::convert::write_json;
use num::rational::Ratio;

/// Example: t[a,b] = 4*X[a,c]*U[c,b] + 2*X[a,c]*V[c,b] - 2*Y[a,c]*U[c,b] - Y[a,c]*V[c,b] + Z[a,b]
///
/// The biclique {X, Y} × {U, V} should be factored out.
/// Factored: (2*X - Y) * (2*U + V) + Z  (or equivalent coefficient split)
fn main() {
    let mut comp = TensorComputation::new();

    let o = comp.add_range(10);

    let r = comp.add_tensor(&[o, o], vec![]);
    let a = comp.add_tensor(&[o, o], vec![]);
    let b = comp.add_tensor(&[o, o], vec![]);
    let c = comp.add_tensor(&[o, o], vec![]);
    let d = comp.add_tensor(&[o, o], vec![]);
    let e = comp.add_tensor(&[o, o], vec![]);
    let f = comp.add_tensor(&[o, o], vec![]);
    let p = comp.add_tensor(&[o, o], vec![]);
    let q = comp.add_tensor(&[o, o], vec![]);
    let x = comp.add_tensor(&[o, o], vec![]);
    let y = comp.add_tensor(&[o, o], vec![]);

    let i = IndexId(0);
    let j = IndexId(1);
    let k = IndexId(2);
    let l = IndexId(3);
    let m = IndexId(4);
    let n = IndexId(5);

    let ext = vec![
        Index { id: i, range: o },
        Index { id: j, range: o },
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
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![Index {id: k, range: o}, Index]
            }
            make_term(4, x, u),   //  4 * X[a,c] * U[c,b]
            make_term(2, x, v),   //  2 * X[a,c] * V[c,b]
            make_term(-2, y, u),  // -2 * Y[a,c] * U[c,b]
            make_term(-1, y, v),  // -1 * Y[a,c] * V[c,b]
            Term {
                coeff: Ratio::from_integer(1),
                sum_indices: vec![],
                factors: vec![Factor { tensor: z, indices: vec![a, b] }],
            },
        ],
    );

    println!("=== Before Optimization ===");
    println!("{comp}");
    let cost_before = total_cost(&comp);
    println!("Cost: {cost_before}");

    let n = greedy_optimize(&mut comp);

    println!("\n=== After Optimization ({n} factorization(s) applied) ===");
    println!("{comp}");
    let cost_after = total_cost(&comp);
    println!("Cost: {cost_after}");
    println!("Saving: {}", cost_before as i64 - cost_after as i64);

    write_json(&comp, Path::new("factorized.json")).unwrap();
}
