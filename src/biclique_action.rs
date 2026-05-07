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

        let candidate_templates: Vec<Factorization> = candidates
            .iter()
            .map(|candidate| export_candidate_template(def, candidate, left_tid, right_tid))
            .collect();

        return Some(ActionSpace {
            def_index,
            candidate_templates,
            candidates,
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
    let record = candidate_record(space, decision.candidate_index)?;
    let (left_tid, right_tid) = fresh_rewrite_tensor_ids(comp);
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
            candidates.push(CandidateRecord {
                graph: graph.clone(),
                biclique,
            });
        }
    }

    candidates
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
    let source = &source_nodes[node_id];
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
    let id_to_range: HashMap<IndexId, _> = def
        .terms
        .iter()
        .flat_map(|term| term.sum_indices.iter().map(|index| (index.id, index.range)))
        .collect();

    match (
        biclique.left_node_ids.first().copied(),
        biclique.right_node_ids.first().copied(),
    ) {
        (Some(left_id), Some(right_id)) => contracted_indices_for_pair(
            &graph.left_nodes[left_id],
            &graph.right_nodes[right_id],
            &id_to_range,
        ),
        _ => Vec::new(),
    }
}

fn side_external_indices(def: &TensorDef, graph: &ConstrGraph) -> (Vec<Index>, Vec<Index>) {
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
    let contracted_ids: HashSet<IndexId> = contracted.iter().map(|index| index.id).collect();
    let ext_indices: Vec<Index> = side_ext.iter().chain(contracted.iter()).copied().collect();

    let terms: Vec<Term> = biclique_node_ids
        .iter()
        .zip(biclique_coeffs.iter())
        .map(|(&node_id, coeff)| build_side_term(source_nodes, node_id, coeff, &contracted_ids))
        .collect();

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

    contracted
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

fn bits_to_vec(mut mask: u64) -> Vec<usize> {
    let mut out = Vec::new();
    while mask != 0 {
        out.push(mask.trailing_zeros() as usize);
        mask &= mask - 1;
    }
    out
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
    fn test_build_rewrite_from_decision_preserves_candidate_order_for_masks() {
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

        let record = CandidateRecord {
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
        };

        let (left_tid, right_tid) = fresh_rewrite_tensor_ids(&comp);
        let template = export_candidate_template(&source_def, &record, left_tid, right_tid);
        assert_eq!(template.left_definition.terms.len(), 2);
        assert_eq!(template.left_definition.terms[0].factors[0].tensor, high);
        assert_eq!(template.left_definition.terms[1].factors[0].tensor, low);

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
            .expect("mask should select the first candidate left term");

        assert_eq!(
            rewrite.factorization.left_definition.terms,
            vec![template.left_definition.terms[0].clone()]
        );
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
