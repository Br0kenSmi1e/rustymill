use num::rational::Ratio;

use rustymill::biclique::{build_graphs_from_canon_splits, ConstrGraph, GraphEdge};
use rustymill::repr::*;
use rustymill::rl_canon::CanonSplitPair;
use rustymill::rl_parenth::{LastStepIndices, TermSplit};

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

#[test]
fn test_build_graphs_from_canon_splits_returns_two_owner_graphs() {
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

    let graphs = build_graphs_from_canon_splits(
        &simple_def(2),
        &[vec![owner_left_0], vec![owner_left_1]],
    );

    assert_eq!(graphs.len(), 2);
    assert!(graphs.iter().all(|graph| graph.last_step == last_step));
    assert!(graphs.iter().all(|graph| graph.edges.len() == 2));
}

#[test]
fn test_build_graphs_from_canon_splits_keeps_left_and_right_nodes_independent() {
    let last_step = LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    };

    let same = term(1, 1, &[index(2, 0)], vec![factor(7, &[0, 2])]);
    let other = term(1, 1, &[index(2, 0)], vec![factor(8, &[2, 1])]);

    let graphs = build_graphs_from_canon_splits(
        &simple_def(2),
        &[
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
    );

    assert!(graphs.iter().all(|graph| !graph.left_nodes.is_empty()));
    assert!(graphs.iter().all(|graph| !graph.right_nodes.is_empty()));
}
