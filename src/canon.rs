use std::collections::HashMap;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::repr::{
    Factor, IndexId, Rational, RangeId, SymAction, SymGenerator, Index, TensorDef, TensorId, TensorInfo, Term,
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

// ---------------------------------------------------------------------------
// Canonical pool and sub-term canonicalization
// ---------------------------------------------------------------------------

pub fn build_canon_pool(def: &TensorDef) -> HashMap<RangeId, Vec<IndexId>> {
    let mut pool: HashMap<RangeId, Vec<IndexId>> = HashMap::new();
    for term in &def.terms {
        for idx in &term.sum_indices {
            pool.entry(idx.range).or_default().push(idx.id);
        }
    }
    for v in pool.values_mut() {
        v.sort();
        v.dedup();
    }
    pool
}

fn enumerate_sym_group(generators: &[SymGenerator], n: usize) -> Vec<(Vec<usize>, SymAction)> {
    let identity: Vec<usize> = (0..n).collect();
    let mut elements: Vec<(Vec<usize>, SymAction)> = vec![(identity.clone(), SymAction::Identity)];
    let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
    seen.insert(identity);
    let mut qi = 0;
    while qi < elements.len() {
        let (perm, action) = elements[qi].clone();
        qi += 1;
        for gen in generators {
            let new_perm: Vec<usize> = gen.perm.iter().map(|&g| perm[g]).collect();
            if seen.insert(new_perm.clone()) {
                elements.push((new_perm, action.combine(gen.action)));
            }
        }
    }
    elements
}

fn term_lt(a: &Term, b: &Term) -> bool {
    let ac = a.coeff;
    let bc = b.coeff;
    if ac != bc {
        let lhs = (*ac.numer() as i128) * (*bc.denom() as i128);
        let rhs = (*bc.numer() as i128) * (*ac.denom() as i128);
        if lhs != rhs { return lhs < rhs; }
    }
    for (fa, fb) in a.factors.iter().zip(b.factors.iter()) {
        if fa.tensor.0 != fb.tensor.0 { return fa.tensor.0 < fb.tensor.0; }
        for (ia, ib) in fa.indices.iter().zip(fb.indices.iter()) {
            if ia.0 != ib.0 { return ia.0 < ib.0; }
        }
        if fa.indices.len() != fb.indices.len() { return fa.indices.len() < fb.indices.len(); }
    }
    a.factors.len() < b.factors.len()
}

fn next_permutation(perm: &mut Vec<usize>) -> bool {
    let n = perm.len();
    if n <= 1 { return false; }
    let mut i = n - 1;
    while i > 0 && perm[i - 1] >= perm[i] { i -= 1; }
    if i == 0 { return false; }
    let pivot = i - 1;
    let mut j = n - 1;
    while perm[j] <= perm[pivot] { j -= 1; }
    perm.swap(pivot, j);
    perm[i..].reverse();
    true
}

fn advance_group_perms(group_perms: &mut Vec<Vec<usize>>) -> bool {
    for gp in group_perms.iter_mut().rev() {
        if next_permutation(gp) { return true; }
        let n = gp.len();
        *gp = (0..n).collect();
    }
    false
}

pub fn canonicalize_sub_term(
    sub: &Term,
    ext_ids: &HashSet<IndexId>,
    ext_range: &HashMap<IndexId, RangeId>,
    contracted_ids: &HashSet<IndexId>,
    fix_contracted: Option<&HashMap<IndexId, IndexId>>,
    is_left: bool,
    pool: &HashMap<RangeId, Vec<IndexId>>,
    tensors: &[TensorInfo],
) -> (Term, HashMap<IndexId, IndexId>) {
    let mut dummy_range: HashMap<IndexId, RangeId> = HashMap::new();
    for idx in &sub.sum_indices {
        dummy_range.insert(idx.id, idx.range);
    }

    let range_of = |id: IndexId| -> RangeId {
        if let Some(&r) = ext_range.get(&id) { r }
        else { dummy_range.get(&id).copied().unwrap_or(RangeId(u32::MAX)) }
    };

    let mut contracted_count: HashMap<RangeId, usize> = HashMap::new();
    // Always compute contracted_count for left-side terms so non-contracted
    // dummies get the correct pool offset (after contracted slots).
    if fix_contracted.is_none() || is_left {
        for idx in &sub.sum_indices {
            if contracted_ids.contains(&idx.id) {
                *contracted_count.entry(idx.range).or_insert(0) += 1;
            }
        }
    }

    let sym_groups: Vec<Vec<(Vec<usize>, SymAction)>> = sub.factors.iter()
        .map(|f| enumerate_sym_group(&tensors[f.tensor.0 as usize].symmetry, f.indices.len()))
        .collect();

    let combo_sizes: Vec<usize> = sym_groups.iter().map(|g| g.len()).collect();
    let mut best_term: Option<Term> = None;
    let mut best_contracted: HashMap<IndexId, IndexId> = HashMap::new();

    let mut combo = vec![0usize; sub.factors.len()];
    loop {
        let coeff_action = sub.factors.iter().enumerate()
            .fold(SymAction::Identity, |acc, (i, _)| acc.combine(sym_groups[i][combo[i]].1));
        let coeff = apply_action_to_coeff(sub.coeff, coeff_action);

        let mut perm_factors: Vec<(TensorId, Vec<IndexId>)> = sub.factors.iter().enumerate()
            .map(|(i, f)| {
                let (perm, _) = &sym_groups[i][combo[i]];
                let indices: Vec<IndexId> = perm.iter().map(|&p| f.indices[p]).collect();
                (f.tensor, indices)
            })
            .collect();

        perm_factors.sort_by_key(|(tid, indices)| {
            let rs: Vec<RangeId> = indices.iter().map(|&id| range_of(id)).collect();
            (*tid, rs)
        });

        // Find tied groups
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut i = 0;
        while i < perm_factors.len() {
            let key = {
                let (tid, indices) = &perm_factors[i];
                let rs: Vec<RangeId> = indices.iter().map(|&id| range_of(id)).collect();
                (*tid, rs)
            };
            let mut j = i + 1;
            while j < perm_factors.len() {
                let key2 = {
                    let (tid, indices) = &perm_factors[j];
                    let rs: Vec<RangeId> = indices.iter().map(|&id| range_of(id)).collect();
                    (*tid, rs)
                };
                if key != key2 { break; }
                j += 1;
            }
            groups.push((i..j).collect());
            i = j;
        }

        let mut group_perms: Vec<Vec<usize>> = groups.iter()
            .map(|g| (0..g.len()).collect())
            .collect();

        loop {
            let ordering: Vec<(TensorId, Vec<IndexId>)> = groups.iter().zip(group_perms.iter())
                .flat_map(|(g, gp)| gp.iter().map(|&p| perm_factors[g[p]].clone()))
                .collect();

            let mut remap: HashMap<IndexId, IndexId> = HashMap::new();
            let mut contracted_map: HashMap<IndexId, IndexId> = HashMap::new();
            let mut c_counter: HashMap<RangeId, usize> = HashMap::new();
            let mut li_counter: HashMap<RangeId, usize> = HashMap::new();
            let mut ri_counter: HashMap<RangeId, usize> = HashMap::new();

            for (_, indices) in &ordering {
                for &id in indices {
                    if remap.contains_key(&id) { continue; }
                    if ext_ids.contains(&id) {
                        remap.insert(id, id);
                    } else if contracted_ids.contains(&id) {
                        if let Some(fixed) = fix_contracted {
                            if let Some(&cid) = fixed.get(&id) {
                                remap.insert(id, cid);
                            }
                        } else {
                            let range = dummy_range[&id];
                            let slot = *c_counter.entry(range).or_insert(0);
                            c_counter.insert(range, slot + 1);
                            let canonical = pool[&range][slot];
                            remap.insert(id, canonical);
                            contracted_map.insert(id, canonical);
                        }
                    } else {
                        let range = dummy_range[&id];
                        let p = pool.get(&range).map(|v| v.as_slice()).unwrap_or(&[]);
                        let canonical = if is_left {
                            let offset = contracted_count.get(&range).copied().unwrap_or(0);
                            let slot = *li_counter.entry(range).or_insert(0);
                            li_counter.insert(range, slot + 1);
                            p[offset + slot]
                        } else {
                            let slot = *ri_counter.entry(range).or_insert(0);
                            ri_counter.insert(range, slot + 1);
                            p[p.len() - 1 - slot]
                        };
                        remap.insert(id, canonical);
                    }
                }
            }

            let new_factors: Vec<Factor> = ordering.iter()
                .map(|(tid, indices)| Factor {
                    tensor: *tid,
                    indices: indices.iter().map(|&id| remap[&id]).collect(),
                })
                .collect();
            let new_sum_indices: Vec<Index> = sub.sum_indices.iter()
                .filter(|idx| !contracted_ids.contains(&idx.id))
                .map(|idx| Index { id: remap[&idx.id], range: idx.range })
                .collect();
            let candidate = Term { coeff, sum_indices: new_sum_indices, factors: new_factors };

            if best_term.is_none() || term_lt(&candidate, best_term.as_ref().unwrap()) {
                best_term = Some(candidate);
                if fix_contracted.is_none() {
                    best_contracted = contracted_map;
                }
            }

            if !advance_group_perms(&mut group_perms) { break; }
        }

        // Advance combo
        let mut carry = true;
        for k in (0..combo.len()).rev() {
            if carry {
                combo[k] += 1;
                if combo[k] >= combo_sizes[k] {
                    combo[k] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry { break; }
    }

    (best_term.unwrap(), best_contracted)
}
