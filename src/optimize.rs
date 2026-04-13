use crate::constr::{factorizations, Factorization};
use crate::parenth::parenthesize;
use crate::repr::{TensorComputation, TensorDef};

/// Apply a factorization to a TensorComputation.
///
/// Replaces the definition at `def_index` with the factored version:
/// intermediates are inserted before it, consumed terms are removed,
/// and the replacement term is added.
///
/// Also registers the intermediate tensors in the computation.
pub fn apply_factorization(
    comp: &mut TensorComputation,
    def_index: usize,
    fact: &Factorization,
) {
    // Register intermediate tensors
    for interm in &fact.intermediates {
        let slots: Vec<_> = interm.ext_indices.iter().map(|idx| idx.range).collect();
        comp.add_tensor(&slots, vec![]);
    }

    // Modify the target definition: remove consumed terms, add replacement
    let def = &mut comp.definitions_mut()[def_index];
    let mut new_terms: Vec<_> = def
        .terms
        .iter()
        .enumerate()
        .filter(|(i, _)| !fact.terms_consumed.contains(i))
        .map(|(_, t)| t.clone())
        .collect();
    new_terms.push(fact.replacement_term.clone());
    def.terms = new_terms;

    // Insert intermediate definitions before the target definition
    let intermediates: Vec<TensorDef> = fact.intermediates.clone();
    for (i, interm) in intermediates.into_iter().enumerate() {
        comp.definitions_mut().insert(def_index + i, interm);
    }
}

/// Find the leftmost TensorDef (starting from `start_from`) that has a
/// profitable factorization, and return its index along with the best one.
fn next_decision(
    comp: &TensorComputation,
    start_from: usize,
) -> Option<(usize, Factorization)> {
    for (i, def) in comp.definitions().iter().enumerate().skip(start_from) {
        if def.terms.len() < 2 {
            continue;
        }

        let prs: Vec<_> = def
            .terms
            .iter()
            .map(|t| parenthesize(t, &def.ext_indices, comp.ranges()))
            .collect();

        let next_id = comp.next_tensor_id();
        let facts = factorizations(def, &prs, comp, next_id);

        if let Some(best) = facts.into_iter().filter(|f| f.saving > 0).max_by_key(|f| f.saving) {
            return Some((i, best));
        }
    }
    None
}

/// Greedy optimization: repeatedly find and apply the best factorization
/// for the leftmost TensorDef with profitable bicliques, until none remain.
///
/// Returns the number of factorizations applied.
pub fn greedy_optimize(comp: &mut TensorComputation) -> usize {
    let mut count = 0;
    let mut start_from = 0;

    while let Some((def_idx, best)) = next_decision(comp, start_from) {
        apply_factorization(comp, def_idx, &best);
        count += 1;
        start_from = def_idx;
    }

    count
}
