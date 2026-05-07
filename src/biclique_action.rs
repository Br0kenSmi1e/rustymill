use std::collections::{HashMap, HashSet};

use num::rational::Ratio;

use crate::biclique::{build_graphs_from_canon_splits, enumerate_bicliques, Biclique, ConstrGraph};
use crate::repr::{Factor, Index, IndexId, TensorComputation, TensorDef, TensorId, Term};
use crate::rl_canon::{build_canon_def_context, canon_split};
use crate::rl_parenth::enumerate_splits;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Factorization {
    pub left_definition: TensorDef,
    pub right_definition: TensorDef,
    pub rewritten_definition: TensorDef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpace {
    pub def_index: usize,
    pub candidate_templates: Vec<Factorization>,
    candidates: Vec<CandidateRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub candidate_index: usize,
    pub left_mask: Vec<bool>,
    pub right_mask: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorizationRewrite {
    pub def_index: usize,
    pub factorization: Factorization,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CandidateRecord {
    graph: ConstrGraph,
    biclique: Biclique,
}

pub fn next_action_space(comp: &TensorComputation, start_from: usize) -> Option<ActionSpace> {
    let (left_tid, right_tid) = fresh_rewrite_tensor_ids(comp);

    for (def_index, def) in comp.definitions().iter().enumerate().skip(start_from) {
        if def.terms.len() < 2 {
            continue;
        }

        let candidates = enumerate_candidate_records(comp, def);
        if candidates.is_empty() {
            continue;
        }

        let mut valid_candidates = Vec::new();
        let mut candidate_templates = Vec::new();
        for candidate in candidates {
            if validate_candidate_record_against_definition(def, &candidate).is_err() {
                continue;
            }
            candidate_templates.push(export_candidate_template(def, &candidate, left_tid, right_tid));
            valid_candidates.push(candidate);
        }
        if valid_candidates.is_empty() {
            continue;
        }

        return Some(ActionSpace {
            def_index,
            candidate_templates,
            candidates: valid_candidates,
        });
    }

    None
}

fn candidate_template(
    space: &ActionSpace,
    candidate_index: usize,
) -> Result<&Factorization, String> {
    space
        .candidate_templates
        .get(candidate_index)
        .ok_or_else(|| {
            format!(
                "candidate_index {} out of range for {} candidates",
                candidate_index,
                space.candidate_templates.len()
            )
        })
}

fn validate_mask_lengths(template: &Factorization, decision: &Decision) -> Result<(), String> {
    let expected_left = template.left_definition.terms.len();
    if decision.left_mask.len() != expected_left {
        return Err(format!(
            "left mask length mismatch: expected {}, got {}",
            expected_left,
            decision.left_mask.len()
        ));
    }

    let expected_right = template.right_definition.terms.len();
    if decision.right_mask.len() != expected_right {
        return Err(format!(
            "right mask length mismatch: expected {}, got {}",
            expected_right,
            decision.right_mask.len()
        ));
    }

    Ok(())
}

fn validate_nonempty_masks(decision: &Decision) -> Result<(), String> {
    if !decision.left_mask.iter().any(|selected| *selected) {
        return Err("left side cannot be empty".to_string());
    }

    if !decision.right_mask.iter().any(|selected| *selected) {
        return Err("right side cannot be empty".to_string());
    }

    Ok(())
}

pub fn validate_decision(space: &ActionSpace, decision: &Decision) -> Result<(), String> {
    let template = candidate_template(space, decision.candidate_index)?;
    validate_mask_lengths(template, decision)?;
    validate_nonempty_masks(decision)?;
    Ok(())
}

fn target_definition(comp: &TensorComputation, def_index: usize) -> Result<&TensorDef, String> {
    comp.definitions().get(def_index).ok_or_else(|| {
        format!(
            "def_index {} out of range for {} definitions",
            def_index,
            comp.definitions().len()
        )
    })
}

fn candidate_record(
    space: &ActionSpace,
    candidate_index: usize,
) -> Result<&CandidateRecord, String> {
    space.candidates.get(candidate_index).ok_or_else(|| {
        format!(
            "candidate_index {} out of range for {} candidates",
            candidate_index,
            space.candidates.len()
        )
    })
}

fn validate_candidate_record_against_definition(
    def: &TensorDef,
    record: &CandidateRecord,
) -> Result<(), String> {
    validate_side_alignment(
        &record.biclique.left_node_ids,
        &record.biclique.left_coeffs,
        "left",
    )?;
    validate_side_alignment(
        &record.biclique.right_node_ids,
        &record.biclique.right_coeffs,
        "right",
    )?;
    validate_nonempty_biclique(&record.biclique)?;
    validate_mask_references(
        record.graph.last_step.left_ext,
        def.ext_indices.len(),
        "left_ext",
    )?;
    validate_mask_references(
        record.graph.last_step.right_ext,
        def.ext_indices.len(),
        "right_ext",
    )?;
    validate_graph_edges_against_definition(def, &record.graph)?;
    validate_biclique_rectangle_against_definition(def, &record.graph, &record.biclique)?;
    Ok(())
}

fn validate_side_alignment(
    node_ids: &[usize],
    coeffs: &[Ratio<i64>],
    side: &str,
) -> Result<(), String> {
    if node_ids.len() != coeffs.len() {
        return Err(format!(
            "{side} node ids and coeffs length mismatch: {} vs {}",
            node_ids.len(),
            coeffs.len()
        ));
    }
    Ok(())
}

fn validate_nonempty_biclique(biclique: &Biclique) -> Result<(), String> {
    if biclique.left_node_ids.is_empty() {
        return Err("candidate biclique left side is empty".to_string());
    }
    if biclique.right_node_ids.is_empty() {
        return Err("candidate biclique right side is empty".to_string());
    }
    Ok(())
}

fn validate_mask_references(mask: u64, source_len: usize, label: &str) -> Result<(), String> {
    if source_len < u64::BITS as usize && (mask >> source_len) != 0 {
        return Err(format!(
            "{label} mask references external indices out of bounds"
        ));
    }
    Ok(())
}

fn validate_graph_edges_against_definition(
    def: &TensorDef,
    graph: &ConstrGraph,
) -> Result<(), String> {
    for edge in &graph.edges {
        if edge.left_id >= graph.left_nodes.len() {
            return Err(format!(
                "candidate graph edge left_id {} out of bounds for {} left nodes",
                edge.left_id,
                graph.left_nodes.len()
            ));
        }
        if edge.right_id >= graph.right_nodes.len() {
            return Err(format!(
                "candidate graph edge right_id {} out of bounds for {} right nodes",
                edge.right_id,
                graph.right_nodes.len()
            ));
        }
        for term_idx in bits_to_vec(edge.terms_used) {
            if term_idx >= def.terms.len() {
                return Err(format!(
                    "candidate graph references term {} but definition has only {} terms",
                    term_idx,
                    def.terms.len()
                ));
            }
        }
    }
    Ok(())
}

fn validate_biclique_rectangle_against_definition(
    def: &TensorDef,
    graph: &ConstrGraph,
    biclique: &Biclique,
) -> Result<(), String> {
    let id_to_range: HashMap<IndexId, _> = def
        .terms
        .iter()
        .flat_map(|term| term.sum_indices.iter().map(|index| (index.id, index.range)))
        .collect();

    let mut expected_interface: Option<Vec<Index>> = None;
    for &left_id in &biclique.left_node_ids {
        let left_term = graph.left_nodes.get(left_id).ok_or_else(|| {
            format!(
                "candidate biclique left node id {} out of bounds for {} left nodes",
                left_id,
                graph.left_nodes.len()
            )
        })?;
        for &right_id in &biclique.right_node_ids {
            let right_term = graph.right_nodes.get(right_id).ok_or_else(|| {
                format!(
                    "candidate biclique right node id {} out of bounds for {} right nodes",
                    right_id,
                    graph.right_nodes.len()
                )
            })?;
            if !graph
                .edges
                .iter()
                .any(|edge| edge.left_id == left_id && edge.right_id == right_id)
            {
                return Err(format!(
                    "candidate biclique rectangle is missing edge ({left_id}, {right_id})"
                ));
            }

            let pair_interface = contracted_indices_for_pair(left_term, right_term, &id_to_range);
            if let Some(expected) = &expected_interface {
                if &pair_interface != expected {
                    return Err(
                        "candidate biclique rectangle has inconsistent contracted interface"
                            .to_string(),
                    );
                }
            } else {
                expected_interface = Some(pair_interface);
            }
        }
    }

    Ok(())
}

fn sub_biclique_from_decision(record: &CandidateRecord, decision: &Decision) -> Biclique {
    let left_node_ids: Vec<usize> = record
        .biclique
        .left_node_ids
        .iter()
        .copied()
        .zip(decision.left_mask.iter().copied())
        .filter_map(|(node_id, selected)| selected.then_some(node_id))
        .collect();
    let left_coeffs: Vec<Ratio<i64>> = record
        .biclique
        .left_coeffs
        .iter()
        .cloned()
        .zip(decision.left_mask.iter().copied())
        .filter_map(|(coeff, selected)| selected.then_some(coeff))
        .collect();

    let right_node_ids: Vec<usize> = record
        .biclique
        .right_node_ids
        .iter()
        .copied()
        .zip(decision.right_mask.iter().copied())
        .filter_map(|(node_id, selected)| selected.then_some(node_id))
        .collect();
    let right_coeffs: Vec<Ratio<i64>> = record
        .biclique
        .right_coeffs
        .iter()
        .cloned()
        .zip(decision.right_mask.iter().copied())
        .filter_map(|(coeff, selected)| selected.then_some(coeff))
        .collect();

    let selected_left: HashSet<usize> = left_node_ids.iter().copied().collect();
    let selected_right: HashSet<usize> = right_node_ids.iter().copied().collect();
    let terms_used = record
        .graph
        .edges
        .iter()
        .filter(|edge| {
            selected_left.contains(&edge.left_id) && selected_right.contains(&edge.right_id)
        })
        .fold(0, |terms_used, edge| terms_used | edge.terms_used);

    Biclique {
        left_node_ids,
        right_node_ids,
        left_coeffs,
        right_coeffs,
        terms_used,
    }
}

pub fn build_rewrite_from_decision(
    comp: &TensorComputation,
    space: &ActionSpace,
    decision: &Decision,
) -> Result<FactorizationRewrite, String> {
    let def = target_definition(comp, space.def_index)?;
    validate_decision(space, decision)?;

    let visible_template = candidate_template(space, decision.candidate_index)?;
    let record = candidate_record(space, decision.candidate_index)?;
    validate_candidate_record_against_definition(def, record)?;
    let (left_tid, right_tid) = fresh_rewrite_tensor_ids(comp);
    let expected_template = export_candidate_template(def, record, left_tid, right_tid);
    if *visible_template != expected_template {
        return Err(format!(
            "candidate template {} no longer matches its hidden candidate",
            decision.candidate_index
        ));
    }

    let sub_biclique = sub_biclique_from_decision(record, decision);
    let factorization = build_factorization(def, &record.graph, &sub_biclique, left_tid, right_tid);

    Ok(FactorizationRewrite {
        def_index: space.def_index,
        factorization,
    })
}

pub fn apply_factorization_rewrite(
    comp: &mut TensorComputation,
    rewrite: FactorizationRewrite,
) -> Result<(), String> {
    verify_rewrite_tensor_ids(comp, &rewrite)?;
    verify_rewrite_def_index(comp, &rewrite)?;
    verify_rewrite_target_definition(comp, &rewrite)?;
    register_rewrite_tensors(comp);
    replace_definition_with_factorization(comp, rewrite);
    Ok(())
}

fn verify_rewrite_tensor_ids(
    comp: &TensorComputation,
    rewrite: &FactorizationRewrite,
) -> Result<(), String> {
    let (expected_left, expected_right) = fresh_rewrite_tensor_ids(comp);
    let actual_left = rewrite.factorization.left_definition.base;
    let actual_right = rewrite.factorization.right_definition.base;

    if actual_left != expected_left || actual_right != expected_right {
        return Err(format!(
            "rewrite tensor ids mismatch: expected ({}, {}), got ({}, {})",
            expected_left.0, expected_right.0, actual_left.0, actual_right.0
        ));
    }

    Ok(())
}

fn verify_rewrite_def_index(
    comp: &TensorComputation,
    rewrite: &FactorizationRewrite,
) -> Result<(), String> {
    if rewrite.def_index >= comp.definitions().len() {
        return Err(format!(
            "def_index {} out of range for {} definitions",
            rewrite.def_index,
            comp.definitions().len()
        ));
    }

    Ok(())
}

fn verify_rewrite_target_definition(
    comp: &TensorComputation,
    rewrite: &FactorizationRewrite,
) -> Result<(), String> {
    let current = target_definition(comp, rewrite.def_index)?;
    let expected = &rewrite.factorization.rewritten_definition;
    if current.base != expected.base || current.ext_indices != expected.ext_indices {
        return Err(format!(
            "target definition mismatch at index {}: expected base {}, got base {}",
            rewrite.def_index, expected.base.0, current.base.0
        ));
    }

    Ok(())
}

fn register_rewrite_tensors(comp: &mut TensorComputation) {
    comp.add_tensor(vec![]);
    comp.add_tensor(vec![]);
}

fn replace_definition_with_factorization(
    comp: &mut TensorComputation,
    rewrite: FactorizationRewrite,
) {
    let FactorizationRewrite {
        def_index,
        factorization,
    } = rewrite;
    let Factorization {
        left_definition,
        right_definition,
        rewritten_definition,
    } = factorization;

    let definitions = comp.definitions_mut();
    definitions.remove(def_index);
    definitions.insert(def_index, rewritten_definition);
    definitions.insert(def_index, right_definition);
    definitions.insert(def_index, left_definition);
}

fn enumerate_candidate_records(comp: &TensorComputation, def: &TensorDef) -> Vec<CandidateRecord> {
    if def.terms.len() < 2 {
        return Vec::new();
    }

    let cx = build_canon_def_context(def, comp.tensors());
    let canon_splits: Vec<Vec<_>> = def
        .terms
        .iter()
        .map(|term| {
            enumerate_splits(term, def)
                .into_iter()
                .map(|split| canon_split(&split, &cx))
                .collect()
        })
        .collect();

    let mut candidates = Vec::new();
    for graph in build_graphs_from_canon_splits(def, &canon_splits) {
        for biclique in enumerate_bicliques(&graph) {
            candidates.push(normalize_candidate_record(
                def,
                CandidateRecord {
                    graph: graph.clone(),
                    biclique,
                },
            ));
        }
    }

    candidates
}

fn normalize_candidate_record(def: &TensorDef, mut record: CandidateRecord) -> CandidateRecord {
    let contracted_ids: HashSet<IndexId> =
        contracted_indices_for_biclique(def, &record.graph, &record.biclique)
            .into_iter()
            .map(|index| index.id)
            .collect();

    normalize_biclique_side(
        &record.graph.left_nodes,
        &mut record.biclique.left_node_ids,
        &mut record.biclique.left_coeffs,
        &contracted_ids,
    );
    normalize_biclique_side(
        &record.graph.right_nodes,
        &mut record.biclique.right_node_ids,
        &mut record.biclique.right_coeffs,
        &contracted_ids,
    );

    record
}

fn normalize_biclique_side(
    source_nodes: &[Term],
    biclique_node_ids: &mut Vec<usize>,
    biclique_coeffs: &mut Vec<Ratio<i64>>,
    contracted_ids: &HashSet<IndexId>,
) {
    assert_eq!(
        biclique_node_ids.len(),
        biclique_coeffs.len(),
        "biclique node ids and coeffs must stay aligned",
    );

    let mut paired: Vec<(usize, Ratio<i64>, TermSortKey)> = biclique_node_ids
        .iter()
        .copied()
        .zip(biclique_coeffs.iter().cloned())
        .map(|(node_id, coeff)| {
            let term = build_side_term(source_nodes, node_id, &coeff, contracted_ids);
            (node_id, coeff, term_sort_key(&term))
        })
        .collect();
    paired.sort_by(|a, b| a.2.cmp(&b.2));

    *biclique_node_ids = paired.iter().map(|(node_id, _, _)| *node_id).collect();
    *biclique_coeffs = paired.into_iter().map(|(_, coeff, _)| coeff).collect();
}

fn fresh_rewrite_tensor_ids(comp: &TensorComputation) -> (TensorId, TensorId) {
    let left_tid = comp.next_tensor_id();
    let right_tid = TensorId(left_tid.0 + 1);
    (left_tid, right_tid)
}

fn export_candidate_template(
    def: &TensorDef,
    record: &CandidateRecord,
    left_tid: TensorId,
    right_tid: TensorId,
) -> Factorization {
    build_factorization(def, &record.graph, &record.biclique, left_tid, right_tid)
}

fn build_side_term(
    source_nodes: &[Term],
    node_id: usize,
    coeff: &Ratio<i64>,
    contracted_ids: &HashSet<IndexId>,
) -> Term {
    let source = source_nodes
        .get(node_id)
        .expect("biclique node id must be in bounds");
    let mut term = source.clone();
    term.coeff = source.coeff.clone() * coeff.clone();
    term.sum_indices
        .retain(|index| !contracted_ids.contains(&index.id));
    term
}

fn build_factorization(
    def: &TensorDef,
    graph: &ConstrGraph,
    biclique: &Biclique,
    left_tid: TensorId,
    right_tid: TensorId,
) -> Factorization {
    let contracted = contracted_indices_for_biclique(def, graph, biclique);
    let (left_ext, right_ext) = side_external_indices(def, graph);
    let consumed = consumed_term_indices(graph, biclique);
    assert!(
        consumed.iter().all(|&term_idx| term_idx < def.terms.len()),
        "biclique consumed term index must refer to a source term",
    );

    let left_definition = build_side_definition(
        &graph.left_nodes,
        &biclique.left_node_ids,
        &biclique.left_coeffs,
        &left_ext,
        &contracted,
        left_tid,
    );
    let right_definition = build_side_definition(
        &graph.right_nodes,
        &biclique.right_node_ids,
        &biclique.right_coeffs,
        &right_ext,
        &contracted,
        right_tid,
    );
    let rewritten_definition = build_rewritten_definition(
        def,
        &left_definition,
        &right_definition,
        &contracted,
        &consumed,
    );

    Factorization {
        left_definition,
        right_definition,
        rewritten_definition,
    }
}

fn contracted_indices_for_biclique(
    def: &TensorDef,
    graph: &ConstrGraph,
    biclique: &Biclique,
) -> Vec<Index> {
    assert!(
        !biclique.left_node_ids.is_empty(),
        "biclique must contain at least one left node",
    );
    assert!(
        !biclique.right_node_ids.is_empty(),
        "biclique must contain at least one right node",
    );

    let id_to_range: HashMap<IndexId, _> = def
        .terms
        .iter()
        .flat_map(|term| term.sum_indices.iter().map(|index| (index.id, index.range)))
        .collect();

    let mut rectangle_interface: Option<Vec<Index>> = None;
    for &left_id in &biclique.left_node_ids {
        let left_term = graph
            .left_nodes
            .get(left_id)
            .expect("biclique left node id must be in bounds");
        for &right_id in &biclique.right_node_ids {
            let right_term = graph
                .right_nodes
                .get(right_id)
                .expect("biclique right node id must be in bounds");
            assert!(
                graph
                    .edges
                    .iter()
                    .any(|edge| edge.left_id == left_id && edge.right_id == right_id),
                "biclique rectangle must contain every edge",
            );

            let pair_interface = contracted_indices_for_pair(left_term, right_term, &id_to_range);
            match &rectangle_interface {
                None => rectangle_interface = Some(pair_interface),
                Some(expected) => assert_eq!(
                    &pair_interface, expected,
                    "biclique rectangle must have a consistent contracted interface",
                ),
            }
        }
    }

    rectangle_interface.unwrap_or_default()
}

fn side_external_indices(def: &TensorDef, graph: &ConstrGraph) -> (Vec<Index>, Vec<Index>) {
    assert_mask_in_bounds(graph.last_step.left_ext, def.ext_indices.len(), "left_ext");
    assert_mask_in_bounds(
        graph.last_step.right_ext,
        def.ext_indices.len(),
        "right_ext",
    );
    (
        bits_to_indices(graph.last_step.left_ext, &def.ext_indices),
        bits_to_indices(graph.last_step.right_ext, &def.ext_indices),
    )
}

fn consumed_term_indices(_: &ConstrGraph, biclique: &Biclique) -> Vec<usize> {
    bits_to_vec(biclique.terms_used)
}

fn build_side_definition(
    source_nodes: &[Term],
    biclique_node_ids: &[usize],
    biclique_coeffs: &[Ratio<i64>],
    side_ext: &[Index],
    contracted: &[Index],
    tensor: TensorId,
) -> TensorDef {
    assert_eq!(
        biclique_node_ids.len(),
        biclique_coeffs.len(),
        "biclique node ids and coeffs must stay aligned",
    );
    let contracted_ids: HashSet<IndexId> = contracted.iter().map(|index| index.id).collect();
    let ext_indices: Vec<Index> = side_ext.iter().chain(contracted.iter()).copied().collect();

    let mut terms: Vec<Term> = biclique_node_ids
        .iter()
        .zip(biclique_coeffs.iter())
        .map(|(&node_id, coeff)| build_side_term(source_nodes, node_id, coeff, &contracted_ids))
        .collect();
    terms.sort_by_key(term_sort_key);

    TensorDef {
        base: tensor,
        ext_indices,
        terms,
    }
}

fn build_rewritten_definition(
    def: &TensorDef,
    left_def: &TensorDef,
    right_def: &TensorDef,
    contracted: &[Index],
    consumed: &[usize],
) -> TensorDef {
    assert!(
        consumed.iter().all(|&term_idx| term_idx < def.terms.len()),
        "consumed term index must be in bounds",
    );
    let consumed: HashSet<usize> = consumed.iter().copied().collect();
    let replacement = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: contracted.to_vec(),
        factors: vec![
            Factor {
                tensor: left_def.base,
                indices: left_def.ext_indices.iter().map(|index| index.id).collect(),
            },
            Factor {
                tensor: right_def.base,
                indices: right_def.ext_indices.iter().map(|index| index.id).collect(),
            },
        ],
    };

    let mut terms: Vec<Term> = def
        .terms
        .iter()
        .enumerate()
        .filter(|(term_idx, _)| !consumed.contains(term_idx))
        .map(|(_, term)| term.clone())
        .collect();
    terms.push(replacement);

    TensorDef {
        base: def.base,
        ext_indices: def.ext_indices.clone(),
        terms,
    }
}

fn contracted_indices_for_pair(
    left_term: &Term,
    right_term: &Term,
    id_to_range: &HashMap<IndexId, crate::repr::RangeId>,
) -> Vec<Index> {
    let right_ids: HashSet<IndexId> = right_term
        .factors
        .iter()
        .flat_map(|factor| factor.indices.iter().copied())
        .collect();

    let mut seen = HashSet::new();
    let mut contracted = Vec::new();
    for factor in &left_term.factors {
        for &index_id in &factor.indices {
            if right_ids.contains(&index_id) && seen.insert(index_id) {
                if let Some(&range) = id_to_range.get(&index_id) {
                    contracted.push(Index {
                        id: index_id,
                        range,
                    });
                }
            }
        }
    }

    contracted.sort_by_key(|index| (index.range.0, index.id.0));
    contracted
}

fn assert_mask_in_bounds(mask: u64, source_len: usize, label: &str) {
    if source_len < u64::BITS as usize {
        assert_eq!(
            mask >> source_len,
            0,
            "{label} mask references external indices out of bounds",
        );
    }
}

fn bits_to_indices(mut mask: u64, source: &[Index]) -> Vec<Index> {
    let mut out = Vec::new();
    while mask != 0 {
        let bit = mask.trailing_zeros() as usize;
        out.push(
            *source
                .get(bit)
                .expect("bitmask must reference an in-bounds external index"),
        );
        mask &= mask - 1;
    }
    out
}

fn bits_to_vec(mut mask: u64) -> Vec<usize> {
    let mut out = Vec::new();
    while mask != 0 {
        out.push(mask.trailing_zeros() as usize);
        mask &= mask - 1;
    }
    out
}

type TermSortKey = (i64, i64, Vec<(u32, u32)>, Vec<(u32, Vec<u32>)>);

fn term_sort_key(term: &Term) -> TermSortKey {
    (
        *term.coeff.numer(),
        *term.coeff.denom(),
        term.sum_indices
            .iter()
            .map(|index| (index.id.0, index.range.0))
            .collect(),
        term.factors
            .iter()
            .map(|factor| {
                (
                    factor.tensor.0,
                    factor.indices.iter().map(|index| index.0).collect(),
                )
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biclique::GraphEdge;
    use crate::rl_parenth::LastStepIndices;

    fn unit_term(factors: Vec<Factor>) -> Term {
        Term {
            coeff: Ratio::from_integer(1),
            sum_indices: Vec::new(),
            factors,
        }
    }

    #[test]
    fn test_build_rewrite_from_decision_uses_visible_template_term_order_for_masks() {
        let mut comp = TensorComputation::new();
        let occ = comp.add_range(10);
        let target = comp.add_tensor(vec![]);
        let low = comp.add_tensor(vec![]);
        let high = comp.add_tensor(vec![]);
        let shared = comp.add_tensor(vec![]);

        let a = IndexId(0);
        let c = IndexId(1);
        let source_def = TensorDef {
            base: target,
            ext_indices: vec![Index { id: a, range: occ }],
            terms: vec![
                Term {
                    coeff: Ratio::from_integer(1),
                    sum_indices: vec![Index { id: c, range: occ }],
                    factors: vec![
                        Factor {
                            tensor: high,
                            indices: vec![a, c],
                        },
                        Factor {
                            tensor: shared,
                            indices: vec![c],
                        },
                    ],
                },
                Term {
                    coeff: Ratio::from_integer(1),
                    sum_indices: vec![Index { id: c, range: occ }],
                    factors: vec![
                        Factor {
                            tensor: low,
                            indices: vec![a, c],
                        },
                        Factor {
                            tensor: shared,
                            indices: vec![c],
                        },
                    ],
                },
            ],
        };
        comp.add_definition(
            source_def.base,
            source_def.ext_indices.clone(),
            source_def.terms.clone(),
        );

        let record = normalize_candidate_record(
            &source_def,
            CandidateRecord {
                graph: ConstrGraph {
                    last_step: LastStepIndices {
                        left_ext: 0b1,
                        right_ext: 0,
                        sums: vec![occ],
                    },
                    left_nodes: vec![
                        unit_term(vec![Factor {
                            tensor: high,
                            indices: vec![a, c],
                        }]),
                        unit_term(vec![Factor {
                            tensor: low,
                            indices: vec![a, c],
                        }]),
                    ],
                    right_nodes: vec![unit_term(vec![Factor {
                        tensor: shared,
                        indices: vec![c],
                    }])],
                    edges: vec![
                        GraphEdge {
                            left_id: 0,
                            right_id: 0,
                            coeff: Ratio::from_integer(1),
                            terms_used: 0b01,
                        },
                        GraphEdge {
                            left_id: 1,
                            right_id: 0,
                            coeff: Ratio::from_integer(1),
                            terms_used: 0b10,
                        },
                    ],
                },
                biclique: Biclique {
                    left_node_ids: vec![0, 1],
                    right_node_ids: vec![0],
                    left_coeffs: vec![Ratio::from_integer(1), Ratio::from_integer(1)],
                    right_coeffs: vec![Ratio::from_integer(1)],
                    terms_used: 0b11,
                },
            },
        );
        assert_eq!(record.biclique.left_node_ids, vec![1, 0]);

        let (left_tid, right_tid) = fresh_rewrite_tensor_ids(&comp);
        let template = export_candidate_template(&source_def, &record, left_tid, right_tid);
        assert_eq!(template.left_definition.terms.len(), 2);
        assert_eq!(template.left_definition.terms[0].factors[0].tensor, low);

        let space = ActionSpace {
            def_index: 0,
            candidate_templates: vec![template.clone()],
            candidates: vec![record],
        };
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true, false],
            right_mask: vec![true],
        };

        let rewrite = build_rewrite_from_decision(&comp, &space, &decision)
            .expect("mask should select the first visible left term");

        assert_eq!(
            rewrite.factorization.left_definition.terms,
            vec![template.left_definition.terms[0].clone()]
        );
    }

    #[test]
    fn test_build_rewrite_from_decision_rejects_mutated_visible_template() {
        let mut comp = TensorComputation::new();
        let target = comp.add_tensor(vec![]);
        let x = comp.add_tensor(vec![]);
        let y = comp.add_tensor(vec![]);
        let p = comp.add_tensor(vec![]);
        let q = comp.add_tensor(vec![]);
        let occ = comp.add_range(10);
        let virt = comp.add_range(12);

        let a = Index { id: IndexId(0), range: occ };
        let b = Index { id: IndexId(1), range: virt };
        let c = Index { id: IndexId(2), range: virt };

        comp.add_definition(
            target,
            vec![a],
            vec![
                Term {
                    coeff: Ratio::from_integer(1),
                    sum_indices: vec![b, c],
                    factors: vec![
                        Factor {
                            tensor: x,
                            indices: vec![a.id, b.id],
                        },
                        Factor {
                            tensor: p,
                            indices: vec![b.id],
                        },
                    ],
                },
                Term {
                    coeff: Ratio::from_integer(1),
                    sum_indices: vec![b, c],
                    factors: vec![
                        Factor {
                            tensor: x,
                            indices: vec![a.id, b.id],
                        },
                        Factor {
                            tensor: q,
                            indices: vec![b.id],
                        },
                    ],
                },
                Term {
                    coeff: Ratio::from_integer(1),
                    sum_indices: vec![b, c],
                    factors: vec![
                        Factor {
                            tensor: y,
                            indices: vec![a.id, c.id],
                        },
                        Factor {
                            tensor: p,
                            indices: vec![c.id],
                        },
                    ],
                },
                Term {
                    coeff: Ratio::from_integer(1),
                    sum_indices: vec![b, c],
                    factors: vec![
                        Factor {
                            tensor: y,
                            indices: vec![a.id, c.id],
                        },
                        Factor {
                            tensor: q,
                            indices: vec![c.id],
                        },
                    ],
                },
            ],
        );

        let mut space = next_action_space(&comp, 0).expect("fixture should be actionable");
        space.candidate_templates[0].left_definition.terms.pop();

        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true],
            right_mask: vec![true; space.candidate_templates[0].right_definition.terms.len()],
        };

        let err = build_rewrite_from_decision(&comp, &space, &decision)
            .expect_err("mutated visible template should be rejected");
        assert!(err.contains("no longer matches"));
    }

    #[test]
    fn test_apply_factorization_rewrite_rejects_stale_tensor_ids() {
        let mut comp = TensorComputation::new();
        let occ = comp.add_range(10);
        let virt = comp.add_range(12);
        let target = comp.add_tensor(vec![]);
        let x = comp.add_tensor(vec![]);
        let y = comp.add_tensor(vec![]);
        let p = comp.add_tensor(vec![]);
        let q = comp.add_tensor(vec![]);

        let a = Index { id: IndexId(0), range: occ };
        let b = Index { id: IndexId(1), range: virt };
        let c = Index { id: IndexId(2), range: virt };

        comp.add_definition(
            target,
            vec![a],
            vec![
                Term {
                    coeff: Ratio::from_integer(1),
                    sum_indices: vec![b],
                    factors: vec![
                        Factor {
                            tensor: x,
                            indices: vec![a.id, b.id],
                        },
                        Factor {
                            tensor: p,
                            indices: vec![b.id],
                        },
                    ],
                },
                Term {
                    coeff: Ratio::from_integer(1),
                    sum_indices: vec![b, c],
                    factors: vec![
                        Factor {
                            tensor: x,
                            indices: vec![a.id, b.id],
                        },
                        Factor {
                            tensor: q,
                            indices: vec![b.id],
                        },
                    ],
                },
                Term {
                    coeff: Ratio::from_integer(1),
                    sum_indices: vec![b, c],
                    factors: vec![
                        Factor {
                            tensor: y,
                            indices: vec![a.id, c.id],
                        },
                        Factor {
                            tensor: p,
                            indices: vec![c.id],
                        },
                    ],
                },
                Term {
                    coeff: Ratio::from_integer(1),
                    sum_indices: vec![c],
                    factors: vec![
                        Factor {
                            tensor: y,
                            indices: vec![a.id, c.id],
                        },
                        Factor {
                            tensor: q,
                            indices: vec![c.id],
                        },
                    ],
                },
            ],
        );

        let space = next_action_space(&comp, 0).expect("fixture should be actionable");
        let template = &space.candidate_templates[0];
        let decision = Decision {
            candidate_index: 0,
            left_mask: vec![true; template.left_definition.terms.len()],
            right_mask: vec![true; template.right_definition.terms.len()],
        };
        let rewrite = build_rewrite_from_decision(&comp, &space, &decision)
            .expect("rewrite should build before comp changes");

        comp.add_tensor(vec![]);

        let err = apply_factorization_rewrite(&mut comp, rewrite)
            .expect_err("stale rewrite tensor ids should be rejected");
        assert!(err.contains("tensor ids mismatch"));
    }
}
