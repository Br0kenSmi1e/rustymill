use std::collections::HashMap;

use num::rational::Ratio;

use crate::repr::{Rational, TensorDef, Term};
use crate::rl_canon::CanonSplitPair;
use crate::rl_parenth::LastStepIndices;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphEdge {
    pub left_id: usize,
    pub right_id: usize,
    pub coeff: Rational,
    pub terms_used: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstrGraph {
    pub last_step: LastStepIndices,
    pub left_nodes: Vec<Term>,
    pub right_nodes: Vec<Term>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Default)]
struct PendingGraph {
    left_ids: HashMap<Term, usize>,
    right_ids: HashMap<Term, usize>,
    left_nodes: Vec<Term>,
    right_nodes: Vec<Term>,
    edges: Vec<GraphEdge>,
    edge_pos: HashMap<(usize, usize), usize>,
}

pub fn build_graphs_from_canon_splits(
    def: &TensorDef,
    canon_splits: &[Vec<CanonSplitPair>],
) -> Vec<ConstrGraph> {
    assert_eq!(
        canon_splits.len(),
        def.terms.len(),
        "canon_splits must stay aligned with def.terms"
    );

    let mut left_buckets: HashMap<LastStepIndices, PendingGraph> = HashMap::new();
    let mut right_buckets: HashMap<LastStepIndices, PendingGraph> = HashMap::new();

    for (term_idx, term_pairs) in canon_splits.iter().enumerate() {
        let term = &def.terms[term_idx];
        let term_coeff = term.coeff.clone();

        for pair in term_pairs {
            insert_split(
                left_buckets
                    .entry(pair.left_assigned.last_step.clone())
                    .or_default(),
                &pair.left_assigned.left_sub_term,
                &pair.left_assigned.right_sub_term,
                term_idx,
                &term_coeff,
            );
            insert_split(
                right_buckets
                    .entry(pair.right_assigned.last_step.clone())
                    .or_default(),
                &pair.right_assigned.left_sub_term,
                &pair.right_assigned.right_sub_term,
                term_idx,
                &term_coeff,
            );
        }
    }

    let mut graphs: Vec<ConstrGraph> = finalize_graphs(left_buckets)
        .into_iter()
        .chain(finalize_graphs(right_buckets))
        .collect();
    sort_graphs_by_last_step(&mut graphs);
    graphs
}

fn ensure_left_node(graph: &mut PendingGraph, term: &Term) -> usize {
    if let Some(&node_id) = graph.left_ids.get(term) {
        node_id
    } else {
        let node_id = graph.left_nodes.len();
        graph.left_nodes.push(term.clone());
        graph.left_ids.insert(term.clone(), node_id);
        node_id
    }
}

fn ensure_right_node(graph: &mut PendingGraph, term: &Term) -> usize {
    if let Some(&node_id) = graph.right_ids.get(term) {
        node_id
    } else {
        let node_id = graph.right_nodes.len();
        graph.right_nodes.push(term.clone());
        graph.right_ids.insert(term.clone(), node_id);
        node_id
    }
}

fn insert_split(
    graph: &mut PendingGraph,
    left_term: &Term,
    right_term: &Term,
    term_idx: usize,
    term_coeff: &Rational,
) {
    let (left_term, right_term, edge_coeff) =
        normalize_edge_contribution(left_term, right_term, term_coeff);

    let left_id = ensure_left_node(graph, &left_term);
    let right_id = ensure_right_node(graph, &right_term);
    let edge_key = (left_id, right_id);

    if let Some(&edge_idx) = graph.edge_pos.get(&edge_key) {
        merge_edge(&mut graph.edges[edge_idx], term_idx, &edge_coeff);
    } else {
        let mut edge = GraphEdge {
            left_id,
            right_id,
            coeff: Ratio::from_integer(0),
            terms_used: 0,
        };
        merge_edge(&mut edge, term_idx, &edge_coeff);
        graph.edge_pos.insert(edge_key, graph.edges.len());
        graph.edges.push(edge);
    }
}

fn normalize_edge_contribution(
    left_term: &Term,
    right_term: &Term,
    term_coeff: &Rational,
) -> (Term, Term, Rational) {
    let edge_coeff = term_coeff.clone() * left_term.coeff.clone() * right_term.coeff.clone();

    let mut normalized_left = left_term.clone();
    normalized_left.coeff = Ratio::from_integer(1);

    let mut normalized_right = right_term.clone();
    normalized_right.coeff = Ratio::from_integer(1);

    (normalized_left, normalized_right, edge_coeff)
}

fn merge_edge(edge: &mut GraphEdge, term_idx: usize, term_coeff: &Rational) {
    assert!(
        term_idx < u64::BITS as usize,
        "terms_used only supports up to {} terms in the MVP",
        u64::BITS
    );

    let term_bit = 1u64 << term_idx;
    if edge.terms_used & term_bit == 0 {
        edge.coeff += term_coeff.clone();
        edge.terms_used |= term_bit;
    }
}

fn finalize_graphs(buckets: HashMap<LastStepIndices, PendingGraph>) -> Vec<ConstrGraph> {
    buckets
        .into_iter()
        .filter_map(|(last_step, mut pending)| {
            pending
                .edges
                .retain(|edge| edge.coeff != Rational::from_integer(0));

            if pending.edges.len() < 2 {
                None
            } else {
                Some(ConstrGraph {
                    last_step,
                    left_nodes: pending.left_nodes,
                    right_nodes: pending.right_nodes,
                    edges: pending.edges,
                })
            }
        })
        .collect()
}

fn sort_graphs_by_last_step(graphs: &mut [ConstrGraph]) {
    graphs.sort_by(|a, b| {
        a.last_step
            .left_ext
            .cmp(&b.last_step.left_ext)
            .then(a.last_step.right_ext.cmp(&b.last_step.right_ext))
            .then(a.last_step.sums.cmp(&b.last_step.sums))
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SearchNode {
    Left(usize),
    Right(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Delta {
    coeff: Rational,
    terms: u64,
}

fn all_candidates(graph: &ConstrGraph) -> Vec<SearchNode> {
    (0..graph.left_nodes.len())
        .map(SearchNode::Left)
        .chain((0..graph.right_nodes.len()).map(SearchNode::Right))
        .collect()
}

fn initial_subg(graph: &ConstrGraph) -> HashMap<SearchNode, Delta> {
    all_candidates(graph)
        .into_iter()
        .map(|node| {
            (
                node,
                Delta {
                    coeff: Ratio::from_integer(1),
                    terms: 0,
                },
            )
        })
        .collect()
}

fn empty_biclique() -> Biclique {
    Biclique {
        left_node_ids: Vec::new(),
        right_node_ids: Vec::new(),
        left_coeffs: Vec::new(),
        right_coeffs: Vec::new(),
        terms_used: 0,
    }
}

fn edge_between(graph: &ConstrGraph, left_id: usize, right_id: usize) -> Option<&GraphEdge> {
    graph
        .edges
        .iter()
        .find(|edge| edge.left_id == left_id && edge.right_id == right_id)
}

fn overlaps_terms(mask: u64, edge_terms: u64) -> bool {
    mask & edge_terms != 0
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Biclique {
    pub left_node_ids: Vec<usize>,
    pub right_node_ids: Vec<usize>,
    pub left_coeffs: Vec<Rational>,
    pub right_coeffs: Vec<Rational>,
    pub terms_used: u64,
}

pub fn enumerate_bicliques(graph: &ConstrGraph) -> Vec<Biclique> {
    if graph.edges.len() < 2 {
        return Vec::new();
    }

    let mut biclique = empty_biclique();
    let mut cand = all_candidates(graph);
    let mut out = Vec::new();
    let subg = initial_subg(graph);

    expand(graph, &mut biclique, &subg, &mut cand, &mut out);
    out
}

fn expand(
    graph: &ConstrGraph,
    biclique: &mut Biclique,
    subg: &HashMap<SearchNode, Delta>,
    cand: &mut Vec<SearchNode>,
    out: &mut Vec<Biclique>,
) {
    if has_sharing(biclique) && subg.is_empty() {
        out.push(canonicalize_biclique(biclique));
        return;
    }

    let subgq = build_child_frontiers(graph, biclique, subg);
    let curr = sift(biclique, cand, subg, &subgq);

    for q in curr {
        let Some(dq) = subg.get(&q) else { continue };
        let Some(pos) = cand.iter().position(|node| *node == q) else {
            continue;
        };

        let removed = cand.remove(pos);
        let child_subg = subgq.get(&removed).cloned().unwrap_or_default();
        let mut child_cand: Vec<SearchNode> = cand
            .iter()
            .copied()
            .filter(|node| child_subg.contains_key(node))
            .collect();

        push(biclique, removed, dq);
        expand(graph, biclique, &child_subg, &mut child_cand, out);
        pop(biclique, removed, dq);
    }
}

fn sift(
    biclique: &Biclique,
    cand: &[SearchNode],
    subg: &HashMap<SearchNode, Delta>,
    subgq: &HashMap<SearchNode, HashMap<SearchNode, Delta>>,
) -> Vec<SearchNode> {
    if biclique.left_node_ids.is_empty() && biclique.right_node_ids.is_empty() {
        return cand
            .iter()
            .filter(|node| matches!(node, SearchNode::Left(_)))
            .copied()
            .collect();
    }

    if biclique.left_node_ids.len() == 1 && biclique.right_node_ids.is_empty() {
        return cand
            .iter()
            .filter(|node| matches!(node, SearchNode::Right(_)))
            .filter(|node| matches!(subg.get(node), Some(delta) if delta.terms != 0))
            .copied()
            .collect();
    }

    let curr = cand.to_vec();

    let mut best_forbidden = Vec::new();
    let mut best_score = 0usize;
    for &q in &curr {
        let forbidden: Vec<SearchNode> = subgq
            .get(&q)
            .map(|next| next.keys().copied().collect())
            .unwrap_or_default();
        let score = forbidden
            .iter()
            .filter(|node| curr.contains(node))
            .count();
        if score > best_score {
            best_score = score;
            best_forbidden = forbidden;
        }
    }

    curr.into_iter()
        .filter(|node| !best_forbidden.contains(node))
        .collect()
}

fn build_child_frontiers(
    graph: &ConstrGraph,
    biclique: &Biclique,
    subg: &HashMap<SearchNode, Delta>,
) -> HashMap<SearchNode, HashMap<SearchNode, Delta>> {
    let mut out = HashMap::new();

    for (q, dq) in subg {
        let mut child = HashMap::new();
        for (r, dr) in subg {
            if q == r {
                continue;
            }
            if let Some(updated) = update_delta(graph, biclique, *q, dq, *r, dr) {
                child.insert(*r, updated);
            }
        }
        out.insert(*q, child);
    }

    out
}

fn update_delta(
    graph: &ConstrGraph,
    biclique: &Biclique,
    q: SearchNode,
    dq: &Delta,
    r: SearchNode,
    dr: &Delta,
) -> Option<Delta> {
    if matches!(
        (q, r),
        (SearchNode::Left(_), SearchNode::Left(_)) | (SearchNode::Right(_), SearchNode::Right(_))
    ) {
        if dq.terms & dr.terms != 0 {
            return None;
        }
        return Some(dr.clone());
    }

    let (left_id, right_id) = match (q, r) {
        (SearchNode::Left(left_id), SearchNode::Right(right_id)) => (left_id, right_id),
        (SearchNode::Right(right_id), SearchNode::Left(left_id)) => (left_id, right_id),
        _ => unreachable!(),
    };

    let edge = edge_between(graph, left_id, right_id)?;

    if overlaps_terms(dq.terms, dr.terms) {
        return None;
    }
    if overlaps_terms(biclique.terms_used, edge.terms_used) {
        return None;
    }
    if overlaps_terms(dq.terms, edge.terms_used) {
        return None;
    }
    if overlaps_terms(dr.terms, edge.terms_used) {
        return None;
    }

    let q_coeff = dq.coeff.clone();
    let expected = edge.coeff.clone() / q_coeff;

    let mut next = dr.clone();
    if dr.terms == 0 {
        next.coeff = expected;
    } else if dr.coeff != expected {
        return None;
    }
    next.terms |= edge.terms_used;
    Some(next)
}

fn has_sharing(biclique: &Biclique) -> bool {
    biclique.left_node_ids.len() >= 2 || biclique.right_node_ids.len() >= 2
}

fn canonicalize_biclique(biclique: &Biclique) -> Biclique {
    let mut left: Vec<(usize, Rational)> = biclique
        .left_node_ids
        .iter()
        .copied()
        .zip(biclique.left_coeffs.iter().cloned())
        .collect();
    left.sort_by_key(|(id, _)| *id);

    let mut right: Vec<(usize, Rational)> = biclique
        .right_node_ids
        .iter()
        .copied()
        .zip(biclique.right_coeffs.iter().cloned())
        .collect();
    right.sort_by_key(|(id, _)| *id);

    Biclique {
        left_node_ids: left.iter().map(|(id, _)| *id).collect(),
        right_node_ids: right.iter().map(|(id, _)| *id).collect(),
        left_coeffs: left.into_iter().map(|(_, coeff)| coeff).collect(),
        right_coeffs: right.into_iter().map(|(_, coeff)| coeff).collect(),
        terms_used: biclique.terms_used,
    }
}

fn push(biclique: &mut Biclique, node: SearchNode, delta: &Delta) {
    biclique.terms_used |= delta.terms;
    let coeff = delta.coeff.clone();
    match node {
        SearchNode::Left(id) => {
            biclique.left_node_ids.push(id);
            biclique.left_coeffs.push(coeff);
        }
        SearchNode::Right(id) => {
            biclique.right_node_ids.push(id);
            biclique.right_coeffs.push(coeff);
        }
    }
}

fn pop(biclique: &mut Biclique, node: SearchNode, delta: &Delta) {
    biclique.terms_used ^= delta.terms;
    match node {
        SearchNode::Left(_) => {
            biclique.left_node_ids.pop();
            biclique.left_coeffs.pop();
        }
        SearchNode::Right(_) => {
            biclique.right_node_ids.pop();
            biclique.right_coeffs.pop();
        }
    }
}
