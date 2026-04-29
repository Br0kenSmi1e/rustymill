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
