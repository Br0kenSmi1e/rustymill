use rustymill::repr::*;
use rustymill::parenth::*;
use rustymill::constr::*;
use num::rational::Ratio;

/// Example: t[a,b] = 4*X[a,c]*U[c,b] + 2*X[a,c]*V[c,b] - 2*Y[a,c]*U[c,b] - Y[a,c]*V[c,b]
///
/// This has a biclique {X, Y} × {U, V} with coefficients:
///   4 = leading * X_coeff * U_coeff
///   2 = leading * X_coeff * V_coeff
///  -2 = leading * Y_coeff * U_coeff
///  -1 = leading * Y_coeff * V_coeff
///
/// One solution: leading=1, X_coeff=2, Y_coeff=-1, U_coeff=2, V_coeff=1
/// Factored: (2*X[a,c] - Y[a,c]) * (2*U[c,b] + V[c,b])
fn main() {
    let mut comp = TensorComputation::new();

    // Ranges: occ = 10
    let occ = comp.add_range(10);

    // Tensors: X, Y, U, V, t (all [occ, occ])
    let x = comp.add_tensor(&[occ, occ], vec![]); // TensorId(0)
    let y = comp.add_tensor(&[occ, occ], vec![]); // TensorId(1)
    let u = comp.add_tensor(&[occ, occ], vec![]); // TensorId(2)
    let v = comp.add_tensor(&[occ, occ], vec![]); // TensorId(3)
    let t = comp.add_tensor(&[occ, occ], vec![]); // TensorId(4)

    // Indices
    let a = IndexId(0); // ext
    let b = IndexId(1); // ext
    let c = IndexId(2); // sum

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

    let terms = vec![
        make_term(4, x, u),   //  4 * X[a,c] * U[c,b]
        make_term(2, x, v),   //  2 * X[a,c] * V[c,b]
        make_term(-2, y, u),  // -2 * Y[a,c] * U[c,b]
        make_term(-1, y, v),  // -1 * Y[a,c] * V[c,b]
    ];

    let def = TensorDef {
        base: t,
        ext_indices: ext.clone(),
        terms: terms.clone(),
    };

    // Step 1: Parenthesize each term
    println!("=== Parenthesization ===");
    let prs: Vec<ParenthResult> = terms.iter()
        .map(|t| parenthesize(t, &ext, comp.ranges()))
        .collect();

    for (i, pr) in prs.iter().enumerate() {
        let n = pr.info.n_factors;
        let full = (1u64 << n) - 1;
        let interm = &pr.memoir[&full];
        println!(
            "Term {}: {} evals, best_cost = {}",
            i,
            interm.evals.len(),
            interm.best_cost
        );
    }

    // Step 2: Find factorizations
    println!("\n=== Factorizations ===");
    let next_id = TensorId(comp.tensors().len() as u32);
    let facts = factorizations(&def, &prs, &comp, next_id);

    println!("Found {} factorizations", facts.len());
    for (i, f) in facts.iter().enumerate() {
        println!(
            "\nFactorization {}: saving = {}, terms consumed = {:?}",
            i, f.saving, f.terms_consumed
        );
        println!("  {} intermediate(s):", f.intermediates.len());
        for (j, interm) in f.intermediates.iter().enumerate() {
            println!(
                "    Intermediate {}: base = {:?}, ext = {} indices, {} term(s)",
                j,
                interm.base,
                interm.ext_indices.len(),
                interm.terms.len()
            );
            for (k, term) in interm.terms.iter().enumerate() {
                println!(
                    "      Term {}: coeff = {}, {} factor(s)",
                    k,
                    term.coeff,
                    term.factors.len()
                );
            }
        }
        println!(
            "  Replacement: coeff = {}, {} factor(s), {} sum indices",
            f.replacement_term.coeff,
            f.replacement_term.factors.len(),
            f.replacement_term.sum_indices.len()
        );
    }

    // Step 3: Pick best factorization
    if let Some(best) = facts.iter().max_by_key(|f| f.saving) {
        println!("\n=== Best Factorization ===");
        println!("Saving: {}", best.saving);
        println!("Consumes terms: {:?}", best.terms_consumed);
        println!("Creates {} intermediate(s)", best.intermediates.len());

        // Show what the optimized computation looks like
        println!("\n=== Optimized Computation ===");
        for interm in &best.intermediates {
            print!("  {:?}[...] = ", interm.base);
            for (i, term) in interm.terms.iter().enumerate() {
                if i > 0 { print!(" + "); }
                if term.coeff != Ratio::from_integer(1) {
                    print!("{} * ", term.coeff);
                }
                for (j, f) in term.factors.iter().enumerate() {
                    if j > 0 { print!(" * "); }
                    print!("T{:?}[...]", f.tensor);
                }
            }
            println!();
        }

        // Remaining terms
        print!("  {:?}[...] = ", def.base);
        let mut first = true;
        // Replacement term
        if best.replacement_term.coeff != Ratio::from_integer(1) {
            print!("{} * ", best.replacement_term.coeff);
        }
        for (j, f) in best.replacement_term.factors.iter().enumerate() {
            if j > 0 { print!(" * "); }
            print!("T{:?}[...]", f.tensor);
        }
        // Untouched terms
        for (i, term) in def.terms.iter().enumerate() {
            if !best.terms_consumed.contains(&i) {
                print!(" + ");
                if term.coeff != Ratio::from_integer(1) {
                    print!("{} * ", term.coeff);
                }
                for (j, f) in term.factors.iter().enumerate() {
                    if j > 0 { print!(" * "); }
                    print!("T{:?}[...]", f.tensor);
                }
            }
        }
        println!();

        // Cost comparison
        let original_cost = rustymill::cost::def_cost(&def, comp.ranges());
        println!("\n=== Cost ===");
        println!("Original cost: {}", original_cost);
        println!("Saving: {}", best.saving);
    } else {
        println!("\nNo profitable factorizations found.");
    }
}
