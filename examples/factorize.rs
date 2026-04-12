use rustymill::repr::*;
use rustymill::optimize::greedy_optimize;
use rustymill::cost::total_cost;
use num::rational::Ratio;

/// Example: t[a,b] = 4*X[a,c]*U[c,b] + 2*X[a,c]*V[c,b] - 2*Y[a,c]*U[c,b] - Y[a,c]*V[c,b] + Z[a,b]
///
/// The biclique {X, Y} × {U, V} should be factored out.
/// Factored: (2*X - Y) * (2*U + V) + Z  (or equivalent coefficient split)
fn main() {
    let mut comp = TensorComputation::new();

    let occ = comp.add_range(10);

    let x = comp.add_tensor(&[occ, occ], vec![]);
    let y = comp.add_tensor(&[occ, occ], vec![]);
    let u = comp.add_tensor(&[occ, occ], vec![]);
    let v = comp.add_tensor(&[occ, occ], vec![]);
    let t = comp.add_tensor(&[occ, occ], vec![]);
    let z = comp.add_tensor(&[occ, occ], vec![]);

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
}
