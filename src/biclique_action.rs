use std::collections::{HashMap, HashSet};

use num::rational::Ratio;

use crate::biclique::{build_graphs_from_canon_splits, enumerate_bicliques, Biclique, ConstrGraph};
use crate::repr::{
    Factor, Index, IndexId, RangeId, Rational, TensorComputation, TensorDef, Term,
};
use crate::rl_canon::{build_canon_def_context, canon_split, CanonSplitPair};
use crate::rl_parenth::enumerate_splits;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredAction {
    pub biclique_index: usize,
    pub left_mask: Vec<bool>,
    pub right_mask: Vec<bool>,
}

#[derive(Clone, Debug)]
/// Produced by this module; callers select visible candidates by index only,
/// while the private sidecar keeps aligned internal execution metadata.
pub struct ActionDecision {
    def_index: usize,
    candidate_bicliques: Vec<TensorComputation>,
    // Invariant: candidate_bicliques[i] must stay aligned with candidates[i]
    // and candidate_fingerprints[i].
    candidates: Vec<CandidateRecord>,
    candidate_fingerprints: Vec<CandidateFingerprint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorizationRewrite {
    pub def_index: usize,
    pub consumed_term_indices: Vec<usize>,
    pub replacement_definitions: Vec<TensorDef>,
    pub new_tensor_count: usize,
}

#[derive(Clone, Debug)]
struct CandidateRecord {
    graph: ConstrGraph,
    biclique: Biclique,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CandidateFingerprint {
    left_ext: u64,
    right_ext: u64,
    sums: Vec<RangeId>,
    left_nodes: Vec<Term>,
    right_nodes: Vec<Term>,
    left_coeffs: Vec<Rational>,
    right_coeffs: Vec<Rational>,
    rectangle_terms_used: Vec<u64>,
}

impl ActionDecision {
    fn new(
        def_index: usize,
        candidate_bicliques: Vec<TensorComputation>,
        candidates: Vec<CandidateRecord>,
    ) -> Self {
        let candidate_fingerprints: Vec<CandidateFingerprint> =
            candidates.iter().map(candidate_fingerprint).collect();
        assert_eq!(
            candidate_bicliques.len(),
            candidates.len(),
            "candidate templates must stay aligned with candidate sidecar"
        );
        assert_eq!(
            candidate_bicliques.len(),
            candidate_fingerprints.len(),
            "candidate templates must stay aligned with candidate fingerprints"
        );
        Self {
            def_index,
            candidate_bicliques,
            candidates,
            candidate_fingerprints,
        }
    }

    pub fn def_index(&self) -> usize {
        self.def_index
    }

    pub fn candidate_bicliques(&self) -> &[TensorComputation] {
        &self.candidate_bicliques
    }
}

pub fn next_action_decision(
    comp: &TensorComputation,
    start_from: usize,
) -> Option<ActionDecision> {
    for (def_index, def) in comp.definitions().iter().enumerate().skip(start_from) {
        if def.terms.len() < 2 {
            continue;
        }

        let cx = build_canon_def_context(def, comp.tensors());
        let term_splits: Vec<Vec<_>> = def
            .terms
            .iter()
            .map(|term| enumerate_splits(term, def))
            .collect();
        let canon_splits: Vec<Vec<CanonSplitPair>> = term_splits
            .iter()
            .map(|splits| {
                splits
                    .iter()
                    .map(|split| canon_split(split, &cx))
                    .collect()
            })
            .collect();

        let graphs = build_graphs_from_canon_splits(def, &canon_splits);
        let mut candidates = Vec::new();
        let mut templates = Vec::new();

        for graph in graphs {
            for biclique in enumerate_bicliques(&graph) {
                templates.push(export_template(def, comp, &graph, &biclique));
                candidates.push(CandidateRecord { graph: graph.clone(), biclique });
            }
        }

        if !templates.is_empty() {
            return Some(ActionDecision::new(def_index, templates, candidates));
        }
    }

    None
}

fn export_template(
    def: &TensorDef,
    comp: &TensorComputation,
    graph: &ConstrGraph,
    biclique: &Biclique,
) -> TensorComputation {
    let mut template = TensorComputation::new();
    for range in comp.ranges() {
        template.add_range(range.size);
    }
    for tensor in comp.tensors() {
        template.add_tensor(tensor.symmetry.clone());
    }

    let mut id_to_range: HashMap<IndexId, RangeId> = HashMap::new();
    for term in &def.terms {
        for idx in &term.sum_indices {
            id_to_range.insert(idx.id, idx.range);
        }
    }

    let contracted = contracted_indices_for_biclique(graph, biclique, &id_to_range);
    let contracted_ids: HashSet<IndexId> = contracted.iter().map(|idx| idx.id).collect();
    let left_ext = bits_to_indices(graph.last_step.left_ext, &def.ext_indices);
    let right_ext = bits_to_indices(graph.last_step.right_ext, &def.ext_indices);

    let left_terms = side_terms(
        &biclique.left_node_ids,
        &biclique.left_coeffs,
        &graph.left_nodes,
        &contracted_ids,
    );
    let right_terms = side_terms(
        &biclique.right_node_ids,
        &biclique.right_coeffs,
        &graph.right_nodes,
        &contracted_ids,
    );

    let left_id = template.next_tensor_id();
    template.add_tensor(vec![]);
    template.add_definition(
        left_id,
        left_ext.iter().chain(contracted.iter()).cloned().collect(),
        left_terms,
    );

    let right_id = template.next_tensor_id();
    template.add_tensor(vec![]);
    template.add_definition(
        right_id,
        right_ext.iter().chain(contracted.iter()).cloned().collect(),
        right_terms,
    );

    let consumed = bits_to_vec(biclique.terms_used);
    let replacement = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: contracted,
        factors: vec![
            Factor {
                tensor: left_id,
                indices: template.definitions()[0]
                    .ext_indices
                    .iter()
                    .map(|idx| idx.id)
                    .collect(),
            },
            Factor {
                tensor: right_id,
                indices: template.definitions()[1]
                    .ext_indices
                    .iter()
                    .map(|idx| idx.id)
                    .collect(),
            },
        ],
    };

    let mut rewritten_terms: Vec<Term> = def
        .terms
        .iter()
        .enumerate()
        .filter(|(idx, _)| !consumed.contains(idx))
        .map(|(_, term)| term.clone())
        .collect();
    rewritten_terms.insert(0, replacement);
    template.add_definition(def.base, def.ext_indices.clone(), rewritten_terms);

    template
}

fn bits_to_vec(mut mask: u64) -> Vec<usize> {
    let mut out = Vec::new();
    while mask != 0 {
        out.push(mask.trailing_zeros() as usize);
        mask &= mask - 1;
    }
    out
}

fn bits_to_indices(mut mask: u64, source: &[Index]) -> Vec<Index> {
    let mut out = Vec::new();
    while mask != 0 {
        let bit = mask.trailing_zeros() as usize;
        out.push(source[bit]);
        mask &= mask - 1;
    }
    out
}

fn contracted_indices_from_terms(
    left: &Term,
    right: &Term,
    id_to_range: &HashMap<IndexId, RangeId>,
) -> Vec<Index> {
    let right_sum_ids: HashSet<IndexId> = right
        .sum_indices
        .iter()
        .map(|index| index.id)
        .collect();

    left.sum_indices
        .iter()
        .filter(|index| right_sum_ids.contains(&index.id))
        .map(|index| Index {
            id: index.id,
            range: id_to_range[&index.id],
        })
        .collect()
}

fn contracted_indices_for_biclique(
    graph: &ConstrGraph,
    biclique: &Biclique,
    id_to_range: &HashMap<IndexId, RangeId>,
) -> Vec<Index> {
    let mut expected: Option<Vec<Index>> = None;
    let mut expected_signature: Option<Vec<(IndexId, RangeId)>> = None;

    for &left_id in &biclique.left_node_ids {
        for &right_id in &biclique.right_node_ids {
            let edge = graph
                .edges
                .iter()
                .find(|edge| edge.left_id == left_id && edge.right_id == right_id)
                .expect("biclique export requires every rectangle edge to exist");

            assert!(
                edge.terms_used != 0,
                "biclique export requires every rectangle edge to have source provenance"
            );

            let actual = contracted_indices_from_terms(
                &graph.left_nodes[left_id],
                &graph.right_nodes[right_id],
                id_to_range,
            );
            let actual_signature = contracted_signature(&actual);

            if let Some(signature) = &expected_signature {
                assert_eq!(
                    actual_signature, *signature,
                    "inconsistent contracted index signature across canonical biclique edges"
                );
            } else {
                expected_signature = Some(actual_signature);
                expected = Some(actual);
            }
        }
    }

    expected.expect("biclique export requires at least one contributing edge")
}

fn contracted_signature(indices: &[Index]) -> Vec<(IndexId, RangeId)> {
    let mut signature: Vec<(IndexId, RangeId)> =
        indices.iter().map(|idx| (idx.id, idx.range)).collect();
    signature.sort();
    signature
}

fn side_terms(
    node_ids: &[usize],
    coeffs: &[Rational],
    source_nodes: &[Term],
    contracted_ids: &HashSet<IndexId>,
) -> Vec<Term> {
    node_ids
        .iter()
        .zip(coeffs.iter())
        .map(|(node_id, coeff)| {
            let node = &source_nodes[*node_id];
            Term {
                coeff: coeff.clone(),
                sum_indices: node
                    .sum_indices
                    .iter()
                    .filter(|idx| !contracted_ids.contains(&idx.id))
                    .cloned()
                    .collect(),
                factors: node.factors.clone(),
            }
        })
        .collect()
}

pub fn validate_action_selection(
    decision: &ActionDecision,
    action: &StructuredAction,
) -> Result<(), String> {
    let Some(template) = decision.candidate_bicliques.get(action.biclique_index) else {
        return Err("biclique_index out of range".to_string());
    };

    let left_len = template.definitions()[0].terms.len();
    let right_len = template.definitions()[1].terms.len();

    if action.left_mask.len() != left_len {
        return Err("left_mask length mismatch".to_string());
    }
    if action.right_mask.len() != right_len {
        return Err("right_mask length mismatch".to_string());
    }
    if !action.left_mask.iter().any(|keep| *keep) {
        return Err("left_mask keeps no terms".to_string());
    }
    if !action.right_mask.iter().any(|keep| *keep) {
        return Err("right_mask keeps no terms".to_string());
    }

    Ok(())
}

pub fn rewrite_from_action_selection(
    comp: &TensorComputation,
    start_from: usize,
    decision: &ActionDecision,
    action: &StructuredAction,
) -> Result<FactorizationRewrite, String> {
    validate_action_selection(decision, action)?;

    let fresh = next_action_decision(comp, start_from)
        .ok_or_else(|| "no actionable definition exists".to_string())?;
    if fresh.def_index != decision.def_index {
        return Err("stale decision: actionable definition changed".to_string());
    }

    if fresh.candidate_fingerprints != decision.candidate_fingerprints {
        return Err("hidden candidate metadata no longer matches the active definition".to_string());
    }

    let current_def = &comp.definitions()[decision.def_index];
    let record = decision
        .candidates
        .get(action.biclique_index)
        .ok_or_else(|| "biclique_index out of range".to_string())?;

    let selected_left: Vec<usize> = action
        .left_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, keep)| keep.then_some(record.biclique.left_node_ids[idx]))
        .collect();
    let selected_right: Vec<usize> = action
        .right_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, keep)| keep.then_some(record.biclique.right_node_ids[idx]))
        .collect();

    let selected_left_coeffs: Vec<Rational> = action
        .left_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, keep)| keep.then(|| record.biclique.left_coeffs[idx].clone()))
        .collect();
    let selected_right_coeffs: Vec<Rational> = action
        .right_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, keep)| keep.then(|| record.biclique.right_coeffs[idx].clone()))
        .collect();

    let consumed = consumed_rectangle_terms(&record.graph, &selected_left, &selected_right)?;
    let replacement_definitions = rebuild_selected_definitions(
        current_def,
        comp,
        &record.graph,
        &selected_left,
        &selected_right,
        &selected_left_coeffs,
        &selected_right_coeffs,
        &consumed,
    );

    Ok(FactorizationRewrite {
        def_index: decision.def_index,
        consumed_term_indices: consumed,
        new_tensor_count: replacement_definitions.len().saturating_sub(1),
        replacement_definitions,
    })
}

fn candidate_fingerprint(record: &CandidateRecord) -> CandidateFingerprint {
    let rectangle_terms_used = record
        .biclique
        .left_node_ids
        .iter()
        .flat_map(|left_id| {
            record.biclique.right_node_ids.iter().map(move |right_id| {
                record
                    .graph
                    .edges
                    .iter()
                    .find(|edge| edge.left_id == *left_id && edge.right_id == *right_id)
                    .expect("candidate fingerprint requires every rectangle edge to exist")
                    .terms_used
            })
        })
        .collect();

    CandidateFingerprint {
        left_ext: record.graph.last_step.left_ext,
        right_ext: record.graph.last_step.right_ext,
        sums: record.graph.last_step.sums.clone(),
        left_nodes: record
            .biclique
            .left_node_ids
            .iter()
            .map(|idx| record.graph.left_nodes[*idx].clone())
            .collect(),
        right_nodes: record
            .biclique
            .right_node_ids
            .iter()
            .map(|idx| record.graph.right_nodes[*idx].clone())
            .collect(),
        left_coeffs: record.biclique.left_coeffs.clone(),
        right_coeffs: record.biclique.right_coeffs.clone(),
        rectangle_terms_used,
    }
}

fn consumed_rectangle_terms(
    graph: &ConstrGraph,
    left_ids: &[usize],
    right_ids: &[usize],
) -> Result<Vec<usize>, String> {
    let mut terms = Vec::new();

    for left_id in left_ids {
        for right_id in right_ids {
            let edge = graph
                .edges
                .iter()
                .find(|edge| edge.left_id == *left_id && edge.right_id == *right_id)
                .ok_or_else(|| {
                    "hidden candidate metadata no longer matches rectangle".to_string()
                })?;
            terms.extend(bits_to_vec(edge.terms_used));
        }
    }

    terms.sort();
    terms.dedup();
    Ok(terms)
}

fn rebuild_selected_definitions(
    def: &TensorDef,
    comp: &TensorComputation,
    graph: &ConstrGraph,
    left_ids: &[usize],
    right_ids: &[usize],
    left_coeffs: &[Rational],
    right_coeffs: &[Rational],
    consumed: &[usize],
) -> Vec<TensorDef> {
    let mut id_to_range: HashMap<IndexId, RangeId> = HashMap::new();
    for term in &def.terms {
        for idx in &term.sum_indices {
            id_to_range.insert(idx.id, idx.range);
        }
    }

    let contracted = contracted_indices_from_terms(
        &graph.left_nodes[left_ids[0]],
        &graph.right_nodes[right_ids[0]],
        &id_to_range,
    );
    let contracted_ids: HashSet<IndexId> = contracted.iter().map(|idx| idx.id).collect();
    let left_ext = bits_to_indices(graph.last_step.left_ext, &def.ext_indices);
    let right_ext = bits_to_indices(graph.last_step.right_ext, &def.ext_indices);

    let mut next_id = comp.next_tensor_id().0;
    let (left_tid, left_idx_ids, left_def) = materialize_side(
        left_ids,
        left_coeffs,
        &graph.left_nodes,
        &left_ext,
        &contracted,
        &contracted_ids,
        &mut next_id,
    );
    let (right_tid, right_idx_ids, right_def) = materialize_side(
        right_ids,
        right_coeffs,
        &graph.right_nodes,
        &right_ext,
        &contracted,
        &contracted_ids,
        &mut next_id,
    );

    let replacement_term = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: contracted,
        factors: vec![
            Factor {
                tensor: left_tid,
                indices: left_idx_ids,
            },
            Factor {
                tensor: right_tid,
                indices: right_idx_ids,
            },
        ],
    };

    let mut rewritten_terms: Vec<Term> = def
        .terms
        .iter()
        .enumerate()
        .filter(|(idx, _)| !consumed.contains(idx))
        .map(|(_, term)| term.clone())
        .collect();
    rewritten_terms.insert(0, replacement_term);

    vec![
        left_def,
        right_def,
        TensorDef {
            base: def.base,
            ext_indices: def.ext_indices.clone(),
            terms: rewritten_terms,
        },
    ]
}

fn materialize_side(
    node_ids: &[usize],
    coeffs: &[Rational],
    source_nodes: &[Term],
    side_ext: &[Index],
    contracted: &[Index],
    contracted_ids: &HashSet<IndexId>,
    next_id: &mut u32,
) -> (crate::repr::TensorId, Vec<IndexId>, TensorDef) {
    let ext_indices: Vec<Index> = side_ext.iter().chain(contracted.iter()).cloned().collect();
    let ext_index_ids: Vec<IndexId> = ext_indices.iter().map(|idx| idx.id).collect();
    let tensor = crate::repr::TensorId(*next_id);
    *next_id += 1;

    let terms = node_ids
        .iter()
        .zip(coeffs.iter())
        .map(|(node_id, coeff)| {
            let node = &source_nodes[*node_id];
            Term {
                coeff: coeff.clone(),
                sum_indices: node
                    .sum_indices
                    .iter()
                    .filter(|idx| !contracted_ids.contains(&idx.id))
                    .cloned()
                    .collect(),
                factors: node.factors.clone(),
            }
        })
        .collect();

    (
        tensor,
        ext_index_ids,
        TensorDef {
            base: tensor,
            ext_indices,
            terms,
        },
    )
}

pub fn apply_action_selection(
    comp: &mut TensorComputation,
    start_from: usize,
    decision: &ActionDecision,
    action: &StructuredAction,
) -> Result<(), String> {
    let rewrite = rewrite_from_action_selection(comp, start_from, decision, action)?;
    let rewrite = simplify_rewrite_for_apply(rewrite, comp.next_tensor_id());

    register_new_tensors(comp, rewrite.new_tensor_count);

    let def_index = rewrite.def_index;
    comp.definitions_mut().remove(def_index);
    for (offset, def) in rewrite.replacement_definitions.into_iter().enumerate() {
        comp.definitions_mut().insert(def_index + offset, def);
    }

    Ok(())
}

fn simplify_rewrite_for_apply(
    mut rewrite: FactorizationRewrite,
    first_fresh_tensor: crate::repr::TensorId,
) -> FactorizationRewrite {
    if rewrite.replacement_definitions.len() != 3 {
        return rewrite;
    }

    let rewritten = rewrite
        .replacement_definitions
        .pop()
        .expect("rewrite must contain rewritten active definition");
    let right_def = rewrite
        .replacement_definitions
        .pop()
        .expect("rewrite must contain right intermediate definition");
    let left_def = rewrite
        .replacement_definitions
        .pop()
        .expect("rewrite must contain left intermediate definition");

    if let Some(factor) = safe_passthrough_factor(&left_def) {
        let mut rewritten = rewritten;
        replace_intermediate_factor(&mut rewritten, left_def.base, factor);
        rewrite.replacement_definitions = vec![right_def, rewritten];
        rewrite.new_tensor_count = 1;
        renumber_fresh_definitions(
            &mut rewrite.replacement_definitions,
            rewrite.new_tensor_count,
            first_fresh_tensor,
        );
        return rewrite;
    }

    if let Some(factor) = safe_passthrough_factor(&right_def) {
        let mut rewritten = rewritten;
        replace_intermediate_factor(&mut rewritten, right_def.base, factor);
        rewrite.replacement_definitions = vec![left_def, rewritten];
        rewrite.new_tensor_count = 1;
        renumber_fresh_definitions(
            &mut rewrite.replacement_definitions,
            rewrite.new_tensor_count,
            first_fresh_tensor,
        );
        return rewrite;
    }

    rewrite.replacement_definitions = vec![left_def, right_def, rewritten];
    renumber_fresh_definitions(
        &mut rewrite.replacement_definitions,
        rewrite.new_tensor_count,
        first_fresh_tensor,
    );
    rewrite
}

fn safe_passthrough_factor(def: &TensorDef) -> Option<&Factor> {
    if def.terms.len() != 1 {
        return None;
    }

    let term = &def.terms[0];
    if term.coeff != Ratio::from_integer(1) {
        return None;
    }
    if !term.sum_indices.is_empty() {
        return None;
    }
    if term.factors.len() != 1 {
        return None;
    }

    term.factors.first()
}

fn replace_intermediate_factor(def: &mut TensorDef, intermediate: crate::repr::TensorId, factor: &Factor) {
    for term in &mut def.terms {
        for current in &mut term.factors {
            if current.tensor == intermediate {
                current.tensor = factor.tensor;
                current.indices = factor.indices.clone();
            }
        }
    }
}

fn renumber_fresh_definitions(
    defs: &mut [TensorDef],
    fresh_count: usize,
    first_fresh_tensor: crate::repr::TensorId,
) {
    if fresh_count == 0 {
        return;
    }

    let mut remap = HashMap::new();
    for (offset, def) in defs.iter_mut().take(fresh_count).enumerate() {
        let new_id = crate::repr::TensorId(first_fresh_tensor.0 + offset as u32);
        remap.insert(def.base, new_id);
        def.base = new_id;
    }

    for def in defs.iter_mut() {
        for term in &mut def.terms {
            for factor in &mut term.factors {
                if let Some(new_id) = remap.get(&factor.tensor) {
                    factor.tensor = *new_id;
                }
            }
        }
    }
}

fn register_new_tensors(comp: &mut TensorComputation, count: usize) {
    for _ in 0..count {
        comp.add_tensor(vec![]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biclique::GraphEdge;
    use crate::rl_parenth::LastStepIndices;

    fn index(id: u32, range: RangeId) -> Index {
        Index {
            id: IndexId(id),
            range,
        }
    }

    fn factor(tensor: u32, indices: &[u32]) -> Factor {
        Factor {
            tensor: crate::repr::TensorId(tensor),
            indices: indices.iter().copied().map(IndexId).collect(),
        }
    }

    fn term(sum_indices: &[Index], factors: Vec<Factor>) -> Term {
        Term {
            coeff: Ratio::from_integer(1),
            sum_indices: sum_indices.to_vec(),
            factors,
        }
    }

    #[test]
    fn test_contracted_indices_from_terms_only_uses_shared_sum_indices() {
        let occ = RangeId(0);
        let id_to_range = HashMap::from([
            (IndexId(0), occ),
            (IndexId(1), occ),
            (IndexId(2), occ),
        ]);
        let shared_sum = [index(2, occ)];
        let left = term(
            &shared_sum,
            vec![factor(0, &[0, 2]), factor(1, &[2])],
        );
        let right = term(
            &shared_sum,
            vec![factor(2, &[2, 1]), factor(3, &[0])],
        );

        let contracted = contracted_indices_from_terms(&left, &right, &id_to_range);

        assert_eq!(contracted, vec![index(2, occ)]);
    }

    #[test]
    fn test_contracted_indices_for_biclique_panics_on_inconsistent_graph_term_signatures() {
        let occ = RangeId(0);
        let id_to_range = HashMap::from([(IndexId(2), occ), (IndexId(3), occ)]);
        let graph = ConstrGraph {
            last_step: LastStepIndices {
                left_ext: 0b01,
                right_ext: 0b10,
                sums: vec![occ],
            },
            left_nodes: vec![
                term(&[index(2, occ)], vec![factor(0, &[0, 2])]),
                term(&[index(3, occ)], vec![factor(1, &[0, 3])]),
            ],
            right_nodes: vec![
                term(&[index(2, occ)], vec![factor(2, &[2, 1])]),
                term(&[index(3, occ)], vec![factor(3, &[3, 1])]),
            ],
            edges: vec![
                GraphEdge {
                    left_id: 0,
                    right_id: 0,
                    coeff: Ratio::from_integer(1),
                    terms_used: 0b0001,
                },
                GraphEdge {
                    left_id: 0,
                    right_id: 1,
                    coeff: Ratio::from_integer(1),
                    terms_used: 0b0010,
                },
                GraphEdge {
                    left_id: 1,
                    right_id: 0,
                    coeff: Ratio::from_integer(1),
                    terms_used: 0b0100,
                },
                GraphEdge {
                    left_id: 1,
                    right_id: 1,
                    coeff: Ratio::from_integer(1),
                    terms_used: 0b1000,
                },
            ],
        };
        let biclique = Biclique {
            left_node_ids: vec![0, 1],
            right_node_ids: vec![0, 1],
            left_coeffs: vec![Ratio::from_integer(1), Ratio::from_integer(1)],
            right_coeffs: vec![Ratio::from_integer(1), Ratio::from_integer(1)],
            terms_used: 0b1111,
        };

        let result = std::panic::catch_unwind(|| {
            contracted_indices_for_biclique(
                &graph,
                &biclique,
                &id_to_range,
            )
        });

        assert!(result.is_err());
    }
}
