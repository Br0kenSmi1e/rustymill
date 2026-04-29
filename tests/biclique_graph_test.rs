use num::rational::Ratio;

use rustymill::biclique::{build_graphs_from_canon_splits, ConstrGraph, GraphEdge};
use rustymill::repr::*;
use rustymill::rl_canon::CanonSplitPair;
use rustymill::rl_parenth::{LastStepIndices, TermSplit};

struct TwoOwnerGraphsFixture {
    def: TensorDef,
    last_step: LastStepIndices,
    canon_splits: Vec<Vec<CanonSplitPair>>,
    expected_left_nodes: Vec<Vec<Term>>,
    expected_right_nodes: Vec<Vec<Term>>,
}

struct IndependentNodesFixture {
    def: TensorDef,
    last_step: LastStepIndices,
    canon_splits: Vec<Vec<CanonSplitPair>>,
    expected_left_nodes: Vec<Vec<Term>>,
    expected_right_nodes: Vec<Vec<Term>>,
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
        expected_left_nodes: vec![
            vec![left_a.clone(), left_b.clone()],
            vec![right_a.clone(), right_b.clone()],
        ],
        expected_right_nodes: vec![
            vec![right_a, right_b],
            vec![left_a, left_b],
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
        expected_left_nodes: vec![
            vec![same.clone(), other.clone()],
            vec![other.clone(), same.clone()],
        ],
        expected_right_nodes: vec![vec![other.clone(), same.clone()], vec![same, other]],
    }
}

fn assert_edges_match(graph: &ConstrGraph, expected_edges: &[(usize, usize)]) {
    let actual_edges: Vec<(usize, usize)> = graph
        .edges
        .iter()
        .map(|edge: &GraphEdge| (edge.left_id, edge.right_id))
        .collect();
    assert_eq!(actual_edges, expected_edges);
}

#[test]
fn test_build_graphs_from_canon_splits_returns_two_owner_graphs() {
    let fixture = fixture_two_owner_graphs();
    let graphs = build_graphs_from_canon_splits(&fixture.def, &fixture.canon_splits);

    assert_eq!(graphs.len(), 2);
    assert!(graphs.iter().all(|graph| graph.last_step == fixture.last_step));
    assert_eq!(graphs[0].left_nodes, fixture.expected_left_nodes[0].clone());
    assert_eq!(graphs[0].right_nodes, fixture.expected_right_nodes[0].clone());
    assert_eq!(graphs[1].left_nodes, fixture.expected_left_nodes[1].clone());
    assert_eq!(graphs[1].right_nodes, fixture.expected_right_nodes[1].clone());
    assert_edges_match(&graphs[0], &[(0, 0), (1, 1)]);
    assert_edges_match(&graphs[1], &[(0, 0), (1, 1)]);
}

#[test]
fn test_build_graphs_from_canon_splits_keeps_left_and_right_nodes_independent() {
    let fixture = fixture_independent_nodes();
    let graphs = build_graphs_from_canon_splits(&fixture.def, &fixture.canon_splits);

    assert_eq!(graphs.len(), 2);
    assert!(graphs.iter().all(|graph| graph.last_step == fixture.last_step));
    assert_eq!(graphs[0].left_nodes, fixture.expected_left_nodes[0].clone());
    assert_eq!(graphs[0].right_nodes, fixture.expected_right_nodes[0].clone());
    assert_eq!(graphs[1].left_nodes, fixture.expected_left_nodes[1].clone());
    assert_eq!(graphs[1].right_nodes, fixture.expected_right_nodes[1].clone());
    assert_ne!(graphs[0].left_nodes[0], graphs[0].right_nodes[0]);
    assert_ne!(graphs[1].left_nodes[0], graphs[1].right_nodes[0]);
    assert_edges_match(&graphs[0], &[(0, 0), (1, 1)]);
    assert_edges_match(&graphs[1], &[(0, 0), (1, 1)]);
}
