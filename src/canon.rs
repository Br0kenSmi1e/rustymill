use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::repr::{
    IndexId, Rational, RangeId, SymAction, SymGenerator, Index, TensorId, TensorInfo, Term,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CanonFactor {
    pub tensor: TensorId,
    pub indices: Vec<CanonIndex>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CanonIndex {
    pub range: RangeId,
    pub canon_id: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonTerm {
    pub coeff: Rational,
    pub factors: Vec<CanonFactor>,
}

pub fn canon_term(term: &Term, ext_indices: &[Index], tensors: &[TensorInfo]) -> CanonTerm {
    let ext_set: HashMap<IndexId, RangeId> = ext_indices
        .iter()
        .map(|idx| (idx.id, idx.range))
        .collect();
    let dummy_set: HashMap<IndexId, RangeId> = term
        .sum_indices
        .iter()
        .map(|idx| (idx.id, idx.range))
        .collect();

    let mut coeff_action = SymAction::Identity;
    let mut raw_factors: Vec<(TensorId, Vec<IndexId>)> = Vec::with_capacity(term.factors.len());

    for factor in &term.factors {
        let info = &tensors[factor.tensor.0 as usize];
        let (best_indices, action) = canon_factor_indices(&factor.indices, &info.symmetry);
        coeff_action = coeff_action.combine(action);
        raw_factors.push((factor.tensor, best_indices));
    }

    raw_factors.sort();

    let mut dummy_canon: HashMap<IndexId, u32> = HashMap::new();
    let mut range_counters: HashMap<RangeId, u32> = HashMap::new();

    for (_, indices) in &raw_factors {
        for &idx in indices {
            if dummy_set.contains_key(&idx) && !dummy_canon.contains_key(&idx) {
                let range = dummy_set[&idx];
                let counter = range_counters.entry(range).or_insert(0);
                dummy_canon.insert(idx, u32::MAX - *counter);
                *counter += 1;
            }
        }
    }

    let canon_factors: Vec<CanonFactor> = raw_factors
        .into_iter()
        .map(|(tensor, indices)| {
            let canon_indices = indices
                .into_iter()
                .map(|idx| {
                    if let Some(&range) = ext_set.get(&idx) {
                        CanonIndex {
                            range,
                            canon_id: idx.0,
                        }
                    } else {
                        let range = dummy_set[&idx];
                        CanonIndex {
                            range,
                            canon_id: dummy_canon[&idx],
                        }
                    }
                })
                .collect();
            CanonFactor {
                tensor,
                indices: canon_indices,
            }
        })
        .collect();

    let coeff = apply_action_to_coeff(term.coeff, coeff_action);

    CanonTerm {
        coeff,
        factors: canon_factors,
    }
}

fn canon_factor_indices(
    indices: &[IndexId],
    generators: &[SymGenerator],
) -> (Vec<IndexId>, SymAction) {
    if generators.is_empty() {
        return (indices.to_vec(), SymAction::Identity);
    }

    let n = indices.len();
    let identity_perm: Vec<usize> = (0..n).collect();

    let mut elements: Vec<(Vec<usize>, SymAction)> = vec![(identity_perm, SymAction::Identity)];
    let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
    seen.insert((0..n).collect());

    let mut queue_idx = 0;
    while queue_idx < elements.len() {
        let (perm, action) = elements[queue_idx].clone();
        queue_idx += 1;

        for gen in generators {
            let new_perm: Vec<usize> = gen.perm.iter().map(|&g| perm[g]).collect();
            if seen.insert(new_perm.clone()) {
                let new_action = action.combine(gen.action);
                elements.push((new_perm, new_action));
            }
        }
    }

    let mut best_indices = indices.to_vec();
    let mut best_action = SymAction::Identity;

    for (perm, action) in &elements {
        let permuted: Vec<IndexId> = perm.iter().map(|&p| indices[p]).collect();
        if permuted < best_indices {
            best_indices = permuted;
            best_action = *action;
        }
    }

    (best_indices, best_action)
}

fn apply_action_to_coeff(coeff: Rational, action: SymAction) -> Rational {
    match action {
        SymAction::Identity => coeff,
        SymAction::Negate => -coeff,
        SymAction::Conjugate => coeff,
        SymAction::NegateConjugate => -coeff,
    }
}
