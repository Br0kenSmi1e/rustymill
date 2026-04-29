use num::rational::Ratio;

use rustymill::biclique::{build_graphs_from_canon_splits, ConstrGraph, GraphEdge};
use rustymill::repr::*;
use rustymill::rl_canon::CanonSplitPair;
use rustymill::rl_parenth::{LastStepIndices, TermSplit};

struct TwoOwnerGraphsFixture {
    def: TensorDef,
    last_step: LastStepIndices,
    canon_splits: Vec<Vec<CanonSplitPair>>,
    expected_graphs: Vec<ExpectedGraph>,
}

struct IndependentNodesFixture {
    def: TensorDef,
    last_step: LastStepIndices,
    canon_splits: Vec<Vec<CanonSplitPair>>,
    expected_graphs: Vec<ExpectedGraph>,
}

struct ExpectedGraph {
    left_nodes: Vec<Term>,
    right_nodes: Vec<Term>,
    edges: Vec<(usize, usize)>,
}

fn factor(tensor: u32, indices: &[u32]) -> Factor {
    Factor {
        tensor: TensorId(tensor),
        indices: indices.iter().copied().map(IndexId).collect(),
    }
}

fn index(id: u32, range: u32) -> Index {
    Index {
        id: IndexId(id),
        range: RangeId(range),
    }
}

fn term(coeff_num: i64, coeff_den: i64, sum_indices: &[Index], factors: Vec<Factor>) -> Term {
    Term {
        coeff: Ratio::new(coeff_num, coeff_den),
        sum_indices: sum_indices.to_vec(),
        factors,
    }
}

fn split(left: Term, right: Term, last_step: &LastStepIndices) -> TermSplit {
    TermSplit {
        left_sub_term: left,
        right_sub_term: right,
        last_step: last_step.clone(),
    }
}

fn pair(
    left_left: Term,
    left_right: Term,
    right_left: Term,
    right_right: Term,
    last_step: &LastStepIndices,
) -> CanonSplitPair {
    CanonSplitPair {
        left_assigned: split(left_left, left_right, last_step),
        right_assigned: split(right_left, right_right, last_step),
    }
}

fn simple_def(term_count: usize) -> TensorDef {
    TensorDef {
        base: TensorId(0),
        ext_indices: vec![index(0, 0), index(1, 0)],
        terms: (0..term_count)
            .map(|_| term(1, 1, &[], vec![factor(99, &[0, 1])]))
            .collect(),
    }
}

fn fixture_two_owner_graphs() -> TwoOwnerGraphsFixture {
    let last_step = LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    };

    let shared_sum = [index(2, 0)];
    let left_a = term(1, 1, &shared_sum, vec![factor(1, &[0, 2])]);
    let left_b = term(1, 1, &shared_sum, vec![factor(2, &[0, 2])]);
    let right_a = term(1, 1, &shared_sum, vec![factor(3, &[2, 1])]);
    let right_b = term(1, 1, &shared_sum, vec![factor(4, &[2, 1])]);

    let owner_left_0 = pair(
        left_a.clone(),
        right_a.clone(),
        right_a.clone(),
        left_a.clone(),
        &last_step,
    );
    let owner_left_1 = pair(
        left_b.clone(),
        right_b.clone(),
        right_b.clone(),
        left_b.clone(),
        &last_step,
    );

    TwoOwnerGraphsFixture {
        def: simple_def(2),
        last_step,
        canon_splits: vec![vec![owner_left_0], vec![owner_left_1]],
        expected_graphs: vec![
            ExpectedGraph {
                left_nodes: vec![left_a.clone(), left_b.clone()],
                right_nodes: vec![right_a.clone(), right_b.clone()],
                edges: vec![(0, 0), (1, 1)],
            },
            ExpectedGraph {
                left_nodes: vec![right_a.clone(), right_b.clone()],
                right_nodes: vec![left_a.clone(), left_b.clone()],
                edges: vec![(0, 0), (1, 1)],
            },
        ],
    }
}

fn fixture_independent_nodes() -> IndependentNodesFixture {
    let last_step = LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    };

    let same = term(1, 1, &[index(2, 0)], vec![factor(7, &[0, 2])]);
    let other = term(1, 1, &[index(2, 0)], vec![factor(8, &[2, 1])]);

    IndependentNodesFixture {
        def: simple_def(2),
        last_step: last_step.clone(),
        canon_splits: vec![
            vec![pair(
                same.clone(),
                other.clone(),
                other.clone(),
                same.clone(),
                &last_step,
            )],
            vec![pair(
                other.clone(),
                same.clone(),
                same.clone(),
                other.clone(),
                &last_step,
            )],
        ],
        expected_graphs: vec![
            ExpectedGraph {
                left_nodes: vec![same.clone(), other.clone()],
                right_nodes: vec![other.clone(), same.clone()],
                edges: vec![(0, 0), (1, 1)],
            },
            ExpectedGraph {
                left_nodes: vec![other.clone(), same.clone()],
                right_nodes: vec![same, other],
                edges: vec![(0, 0), (1, 1)],
            },
        ],
    }
}

fn graph_edges(graph: &ConstrGraph) -> Vec<(usize, usize)> {
    graph
        .edges
        .iter()
        .map(|edge: &GraphEdge| (edge.left_id, edge.right_id))
        .collect()
}

fn graph_matches_expected(graph: &ConstrGraph, expected: &ExpectedGraph) -> bool {
    graph.left_nodes == expected.left_nodes
        && graph.right_nodes == expected.right_nodes
        && graph_edges(graph) == expected.edges
}

fn assert_graphs_match_unordered(
    graphs: &[ConstrGraph],
    last_step: &LastStepIndices,
    expected_graphs: &[ExpectedGraph],
) {
    assert_eq!(graphs.len(), expected_graphs.len());
    assert!(graphs.iter().all(|graph| graph.last_step == *last_step));

    let mut matched = vec![false; graphs.len()];
    for expected in expected_graphs {
        let found = graphs.iter().enumerate().position(|(index, graph)| {
            !matched[index] && graph_matches_expected(graph, expected)
        });
        let index = found.expect("expected graph shape was not returned");
        matched[index] = true;
    }
}

#[test]
fn test_build_graphs_from_canon_splits_returns_two_owner_graphs() {
    let fixture = fixture_two_owner_graphs();
    let graphs = build_graphs_from_canon_splits(&fixture.def, &fixture.canon_splits);

    assert_graphs_match_unordered(&graphs, &fixture.last_step, &fixture.expected_graphs);
}

#[test]
fn test_build_graphs_from_canon_splits_keeps_left_and_right_nodes_independent() {
    let fixture = fixture_independent_nodes();
    let graphs = build_graphs_from_canon_splits(&fixture.def, &fixture.canon_splits);

    assert_graphs_match_unordered(&graphs, &fixture.last_step, &fixture.expected_graphs);
    assert!(
        graphs
            .iter()
            .all(|graph| graph.left_nodes[0] != graph.right_nodes[0])
    );
}
