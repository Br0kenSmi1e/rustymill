use crate::constr::{factorizations, Factorization};
use crate::parenth::parenthesize;
use crate::repr::{TensorComputation, TensorDef, TensorId};

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

/// Greedy optimization: repeatedly find and apply the best factorization
/// until no more profitable factorizations exist.
///
/// Returns the number of factorizations applied.
pub fn greedy_optimize(comp: &mut TensorComputation) -> usize {
    let mut count = 0;

    loop {
        let mut best_fact: Option<Factorization> = None;
        let mut best_def_idx: usize = 0;
        let mut best_saving: i64 = 0;

        // Search all definitions for the best factorization
        for (def_idx, def) in comp.definitions().iter().enumerate() {
            // Skip definitions with < 2 multi-factor terms
            // (need at least 2 terms to find a biclique)
            if def.terms.len() < 2 {
                continue;
            }

            // Parenthesize each term
            let prs: Vec<_> = def
                .terms
                .iter()
                .map(|t| parenthesize(t, &def.ext_indices, comp.ranges()))
                .collect();

            let next_id = comp.next_tensor_id();
            let facts = factorizations(def, &prs, comp, next_id);

            if let Some(f) = facts.into_iter().max_by_key(|f| f.saving) {
                if f.saving > best_saving {
                    best_saving = f.saving;
                    best_fact = Some(f);
                    best_def_idx = def_idx;
                }
            }
        }

        match best_fact {
            Some(fact) if best_saving > 0 => {
                apply_factorization(comp, best_def_idx, &fact);
                count += 1;
            }
            _ => break,
        }
    }

    count
}
