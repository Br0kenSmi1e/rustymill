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

fn find_graph_by_nodes<'a>(
    graphs: &'a [ConstrGraph],
    left_nodes: &[Term],
    right_nodes: &[Term],
) -> &'a ConstrGraph {
    graphs
        .iter()
        .find(|graph| graph.left_nodes == left_nodes && graph.right_nodes == right_nodes)
        .expect("expected graph orientation was not returned")
}

fn find_edge<'a>(graph: &'a ConstrGraph, left_id: usize, right_id: usize) -> &'a GraphEdge {
    graph.edges
        .iter()
        .find(|edge| edge.left_id == left_id && edge.right_id == right_id)
        .expect("expected edge was not returned")
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
fn test_crate_surface_exposes_biclique_graph_builder_api() {
    let build_fn: fn(&TensorDef, &[Vec<CanonSplitPair>]) -> Vec<ConstrGraph> =
        build_graphs_from_canon_splits;

    let edge = GraphEdge {
        left_id: 0,
        right_id: 1,
        coeff: Ratio::from_integer(3),
        terms_used: 0b101,
    };
    let graph = ConstrGraph {
        last_step: LastStepIndices {
            left_ext: 0b01,
            right_ext: 0b10,
            sums: vec![RangeId(0)],
        },
        left_nodes: vec![],
        right_nodes: vec![],
        edges: vec![edge.clone()],
    };

    assert_eq!(graph.edges, vec![edge]);
    assert!(build_fn(&simple_def(0), &[]).is_empty());
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

#[test]
fn test_build_graphs_from_canon_splits_merges_distinct_terms_on_one_edge() {
    let last_step = LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    };

    let left_shared = term(1, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]);
    let left_other = term(1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]);
    let right_shared = term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]);

    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![index(0, 0), index(1, 0)],
        terms: vec![
            term(2, 1, &[], vec![factor(99, &[0, 1])]),
            term(3, 1, &[], vec![factor(99, &[0, 1])]),
            term(5, 1, &[], vec![factor(99, &[0, 1])]),
        ],
    };

    let graphs = build_graphs_from_canon_splits(
        &def,
        &[
            vec![pair(
                left_shared.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_shared.clone(),
                &last_step,
            )],
            vec![pair(
                left_shared.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_shared.clone(),
                &last_step,
            )],
            vec![pair(
                left_other.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_other.clone(),
                &last_step,
            )],
        ],
    );

    assert_eq!(graphs.len(), 2);

    let left_owner_graph = find_graph_by_nodes(
        &graphs,
        &[left_shared.clone(), left_other.clone()],
        std::slice::from_ref(&right_shared),
    );
    let right_owner_graph = find_graph_by_nodes(
        &graphs,
        std::slice::from_ref(&right_shared),
        &[left_shared.clone(), left_other.clone()],
    );

    let left_owner_shared_edge = find_edge(left_owner_graph, 0, 0);
    assert_eq!(left_owner_shared_edge.coeff, Ratio::from_integer(5));
    assert_eq!(left_owner_shared_edge.terms_used, 0b11);

    let right_owner_shared_edge = find_edge(right_owner_graph, 0, 0);
    assert_eq!(right_owner_shared_edge.coeff, Ratio::from_integer(5));
    assert_eq!(right_owner_shared_edge.terms_used, 0b11);
}

#[test]
fn test_build_graphs_from_canon_splits_ignores_duplicate_same_term_contribution() {
    let last_step = LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    };

    let left_shared = term(1, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]);
    let left_other = term(1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]);
    let right_shared = term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]);

    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![index(0, 0), index(1, 0)],
        terms: vec![
            term(7, 1, &[], vec![factor(99, &[0, 1])]),
            term(5, 1, &[], vec![factor(99, &[0, 1])]),
        ],
    };

    let duplicate = pair(
        left_shared.clone(),
        right_shared.clone(),
        right_shared.clone(),
        left_shared.clone(),
        &last_step,
    );
    let other = pair(
        left_other.clone(),
        right_shared.clone(),
        right_shared.clone(),
        left_other.clone(),
        &last_step,
    );

    let graphs = build_graphs_from_canon_splits(
        &def,
        &[vec![duplicate.clone(), duplicate], vec![other]],
    );

    assert_eq!(graphs.len(), 2);

    let left_owner_graph = find_graph_by_nodes(
        &graphs,
        &[left_shared.clone(), left_other.clone()],
        std::slice::from_ref(&right_shared),
    );
    let right_owner_graph = find_graph_by_nodes(
        &graphs,
        std::slice::from_ref(&right_shared),
        &[left_shared, left_other],
    );

    let left_owner_shared_edge = find_edge(left_owner_graph, 0, 0);
    assert_eq!(left_owner_shared_edge.coeff, Ratio::from_integer(7));
    assert_eq!(left_owner_shared_edge.terms_used, 0b01);

    let right_owner_shared_edge = find_edge(right_owner_graph, 0, 0);
    assert_eq!(right_owner_shared_edge.coeff, Ratio::from_integer(7));
    assert_eq!(right_owner_shared_edge.terms_used, 0b01);
}

#[test]
fn test_build_graphs_from_canon_splits_moves_split_coeffs_to_edges() {
    let last_step = LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    };

    let left_shared = term(2, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]);
    let left_other = term(-1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]);
    let right_shared = term(3, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]);

    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![index(0, 0), index(1, 0)],
        terms: vec![
            term(5, 1, &[], vec![factor(99, &[0, 1])]),
            term(7, 1, &[], vec![factor(99, &[0, 1])]),
        ],
    };

    let graphs = build_graphs_from_canon_splits(
        &def,
        &[
            vec![pair(
                left_shared.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_shared.clone(),
                &last_step,
            )],
            vec![pair(
                left_other.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_other.clone(),
                &last_step,
            )],
        ],
    );

    let normalized_left_shared = term(1, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]);
    let normalized_left_other = term(1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]);
    let normalized_right_shared = term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]);

    let left_owner_graph = find_graph_by_nodes(
        &graphs,
        &[normalized_left_shared.clone(), normalized_left_other.clone()],
        std::slice::from_ref(&normalized_right_shared),
    );
    let right_owner_graph = find_graph_by_nodes(
        &graphs,
        std::slice::from_ref(&normalized_right_shared),
        &[normalized_left_shared.clone(), normalized_left_other.clone()],
    );

    assert_eq!(left_owner_graph.left_nodes[0].coeff, Ratio::from_integer(1));
    assert_eq!(left_owner_graph.left_nodes[1].coeff, Ratio::from_integer(1));
    assert_eq!(left_owner_graph.right_nodes[0].coeff, Ratio::from_integer(1));
    assert_eq!(find_edge(left_owner_graph, 0, 0).coeff, Ratio::from_integer(30));
    assert_eq!(find_edge(left_owner_graph, 1, 0).coeff, Ratio::from_integer(-21));

    assert_eq!(right_owner_graph.left_nodes[0].coeff, Ratio::from_integer(1));
    assert_eq!(right_owner_graph.right_nodes[0].coeff, Ratio::from_integer(1));
    assert_eq!(right_owner_graph.right_nodes[1].coeff, Ratio::from_integer(1));
    assert_eq!(find_edge(right_owner_graph, 0, 0).coeff, Ratio::from_integer(30));
    assert_eq!(find_edge(right_owner_graph, 0, 1).coeff, Ratio::from_integer(-21));
}

#[test]
fn test_build_graphs_from_canon_splits_merges_coeff_only_vertex_variants() {
    let last_step = LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    };

    let left_variant_a = term(2, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]);
    let left_variant_b = term(-3, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]);
    let left_other = term(1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]);
    let right_shared = term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]);

    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![index(0, 0), index(1, 0)],
        terms: vec![
            term(5, 1, &[], vec![factor(99, &[0, 1])]),
            term(7, 1, &[], vec![factor(99, &[0, 1])]),
            term(11, 1, &[], vec![factor(99, &[0, 1])]),
        ],
    };

    let graphs = build_graphs_from_canon_splits(
        &def,
        &[
            vec![pair(
                left_variant_a.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_variant_a.clone(),
                &last_step,
            )],
            vec![pair(
                left_variant_b.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_variant_b.clone(),
                &last_step,
            )],
            vec![pair(
                left_other.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_other.clone(),
                &last_step,
            )],
        ],
    );

    let normalized_left_shared = term(1, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]);
    let normalized_left_other = term(1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]);
    let normalized_right_shared = term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]);

    let left_owner_graph = find_graph_by_nodes(
        &graphs,
        &[normalized_left_shared.clone(), normalized_left_other.clone()],
        std::slice::from_ref(&normalized_right_shared),
    );
    let right_owner_graph = find_graph_by_nodes(
        &graphs,
        std::slice::from_ref(&normalized_right_shared),
        &[normalized_left_shared.clone(), normalized_left_other.clone()],
    );

    assert_eq!(left_owner_graph.left_nodes.len(), 2);
    assert_eq!(right_owner_graph.right_nodes.len(), 2);
    assert_eq!(find_edge(left_owner_graph, 0, 0).coeff, Ratio::from_integer(-11));
    assert_eq!(find_edge(left_owner_graph, 0, 0).terms_used, 0b11);
    assert_eq!(find_edge(right_owner_graph, 0, 0).coeff, Ratio::from_integer(-11));
    assert_eq!(find_edge(right_owner_graph, 0, 0).terms_used, 0b11);
}

#[test]
fn test_build_graphs_from_canon_splits_drops_zero_coeff_edges() {
    let last_step = LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    };

    let left_shared = term(1, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]);
    let left_other = term(1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]);
    let left_third = term(1, 1, &[index(2, 0)], vec![factor(4, &[0, 2])]);
    let right_shared = term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]);
    let right_other = term(1, 1, &[index(2, 0)], vec![factor(5, &[2, 1])]);

    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![index(0, 0), index(1, 0)],
        terms: vec![
            term(1, 1, &[], vec![factor(99, &[0, 1])]),
            term(-1, 1, &[], vec![factor(99, &[0, 1])]),
            term(5, 1, &[], vec![factor(99, &[0, 1])]),
            term(7, 1, &[], vec![factor(99, &[0, 1])]),
        ],
    };

    let graphs = build_graphs_from_canon_splits(
        &def,
        &[
            vec![pair(
                left_shared.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_shared.clone(),
                &last_step,
            )],
            vec![pair(
                left_shared.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_shared.clone(),
                &last_step,
            )],
            vec![pair(
                left_other.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_other.clone(),
                &last_step,
            )],
            vec![pair(
                left_third.clone(),
                right_other.clone(),
                right_other.clone(),
                left_third.clone(),
                &last_step,
            )],
        ],
    );

    assert_eq!(graphs.len(), 2);

    let left_owner_graph = find_graph_by_nodes(
        &graphs,
        &[left_shared.clone(), left_other.clone(), left_third.clone()],
        &[right_shared.clone(), right_other.clone()],
    );
    let right_owner_graph = find_graph_by_nodes(
        &graphs,
        &[right_shared.clone(), right_other.clone()],
        &[left_shared.clone(), left_other.clone(), left_third.clone()],
    );

    assert_eq!(left_owner_graph.edges.len(), 2);
    assert_eq!(find_edge(left_owner_graph, 1, 0).coeff, Ratio::from_integer(5));
    assert_eq!(find_edge(left_owner_graph, 1, 0).terms_used, 0b100);
    assert_eq!(find_edge(left_owner_graph, 2, 1).coeff, Ratio::from_integer(7));
    assert_eq!(find_edge(left_owner_graph, 2, 1).terms_used, 0b1000);
    assert!(
        left_owner_graph
            .edges
            .iter()
            .all(|edge| edge.coeff != Ratio::from_integer(0))
    );

    assert_eq!(right_owner_graph.edges.len(), 2);
    assert_eq!(find_edge(right_owner_graph, 0, 1).coeff, Ratio::from_integer(5));
    assert_eq!(find_edge(right_owner_graph, 0, 1).terms_used, 0b100);
    assert_eq!(find_edge(right_owner_graph, 1, 2).coeff, Ratio::from_integer(7));
    assert_eq!(find_edge(right_owner_graph, 1, 2).terms_used, 0b1000);
    assert!(
        right_owner_graph
            .edges
            .iter()
            .all(|edge| edge.coeff != Ratio::from_integer(0))
    );
}

#[test]
fn test_build_graphs_from_canon_splits_omits_graphs_with_fewer_than_two_edges() {
    let last_step = LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    };

    let left_only = term(1, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]);
    let right_only = term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]);

    let graphs = build_graphs_from_canon_splits(
        &simple_def(1),
        &[vec![pair(
            left_only.clone(),
            right_only.clone(),
            right_only.clone(),
            left_only.clone(),
            &last_step,
        )]],
    );

    assert!(graphs.is_empty());
}

#[test]
fn test_build_graphs_from_canon_splits_omits_graph_when_zero_pruning_leaves_one_edge() {
    let last_step = LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    };

    let left_shared = term(1, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]);
    let left_other = term(1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]);
    let right_shared = term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]);

    let def = TensorDef {
        base: TensorId(0),
        ext_indices: vec![index(0, 0), index(1, 0)],
        terms: vec![
            term(1, 1, &[], vec![factor(99, &[0, 1])]),
            term(-1, 1, &[], vec![factor(99, &[0, 1])]),
            term(5, 1, &[], vec![factor(99, &[0, 1])]),
        ],
    };

    let graphs = build_graphs_from_canon_splits(
        &def,
        &[
            vec![pair(
                left_shared.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_shared.clone(),
                &last_step,
            )],
            vec![pair(
                left_shared.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_shared.clone(),
                &last_step,
            )],
            vec![pair(
                left_other.clone(),
                right_shared.clone(),
                right_shared.clone(),
                left_other.clone(),
                &last_step,
            )],
        ],
    );

    assert!(
        graphs.is_empty(),
        "buckets that start with two edges must be omitted if zero pruning leaves one edge"
    );
}
