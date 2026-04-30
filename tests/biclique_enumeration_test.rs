use num::rational::Ratio;

use rustymill::biclique::{enumerate_bicliques, Biclique, ConstrGraph, GraphEdge};
use rustymill::repr::*;
use rustymill::rl_parenth::LastStepIndices;

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

fn base_last_step() -> LastStepIndices {
    LastStepIndices {
        left_ext: 0b01,
        right_ext: 0b10,
        sums: vec![RangeId(0)],
    }
}

fn graph(
    left_nodes: Vec<Term>,
    right_nodes: Vec<Term>,
    edges: &[(usize, usize, i64, u64)],
) -> ConstrGraph {
    ConstrGraph {
        last_step: base_last_step(),
        left_nodes,
        right_nodes,
        edges: edges
            .iter()
            .map(|(left_id, right_id, coeff, terms_used)| GraphEdge {
                left_id: *left_id,
                right_id: *right_id,
                coeff: Ratio::from_integer(*coeff),
                terms_used: *terms_used,
            })
            .collect(),
    }
}

fn sample_left_nodes() -> Vec<Term> {
    vec![
        term(1, 1, &[index(2, 0)], vec![factor(1, &[0, 2])]),
        term(1, 1, &[index(2, 0)], vec![factor(2, &[0, 2])]),
        term(1, 1, &[index(2, 0)], vec![factor(6, &[0, 2])]),
    ]
}

fn sample_right_nodes() -> Vec<Term> {
    vec![
        term(1, 1, &[index(2, 0)], vec![factor(3, &[2, 1])]),
        term(1, 1, &[index(2, 0)], vec![factor(4, &[2, 1])]),
        term(1, 1, &[index(2, 0)], vec![factor(5, &[2, 1])]),
    ]
}

fn find_biclique<'a>(
    bicliques: &'a [Biclique],
    left_ids: &[usize],
    right_ids: &[usize],
) -> &'a Biclique {
    bicliques
        .iter()
        .find(|biclique| biclique.left_node_ids == left_ids && biclique.right_node_ids == right_ids)
        .expect("expected biclique was not returned")
}

#[test]
fn test_crate_surface_exposes_biclique_enumerator_api() {
    let enumerate_fn: fn(&ConstrGraph) -> Vec<Biclique> = enumerate_bicliques;

    let biclique = Biclique {
        left_node_ids: vec![0],
        right_node_ids: vec![0],
        left_coeffs: vec![Ratio::from_integer(1)],
        right_coeffs: vec![Ratio::from_integer(2)],
        terms_used: 0b1,
    };

    let graph = graph(
        sample_left_nodes(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b1)],
    );

    assert_eq!(biclique.terms_used, 0b1);
    assert!(enumerate_fn(&graph).is_empty());
}

#[test]
fn test_enumerate_bicliques_excludes_trivial_1x1_rectangles() {
    let graph = graph(
        sample_left_nodes()[0..1].to_vec(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b1)],
    );

    assert!(enumerate_bicliques(&graph).is_empty());
}

#[test]
fn test_enumerate_bicliques_bootstraps_to_a_2x1_biclique() {
    let graph = graph(
        sample_left_nodes(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b001), (1, 0, 6, 0b010)],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[0, 1], &[0]);

    assert_eq!(
        biclique.left_coeffs,
        vec![Ratio::from_integer(1), Ratio::from_integer(3)]
    );
    assert_eq!(biclique.right_coeffs, vec![Ratio::from_integer(2)]);
    assert_eq!(biclique.terms_used, 0b011);
}

#[test]
fn test_enumerate_bicliques_finds_factorizable_2x2_rectangle() {
    let graph = graph(
        sample_left_nodes(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, 2, 0b0001),
            (0, 1, 4, 0b0010),
            (1, 0, 6, 0b0100),
            (1, 1, 12, 0b1000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[0, 1], &[0, 1]);

    assert_eq!(
        biclique.left_coeffs,
        vec![Ratio::from_integer(1), Ratio::from_integer(3)]
    );
    assert_eq!(
        biclique.right_coeffs,
        vec![Ratio::from_integer(2), Ratio::from_integer(4)]
    );
    assert_eq!(biclique.terms_used, 0b1111);
}

#[test]
fn test_enumerate_bicliques_rejects_non_factorizable_2x2_rectangle() {
    let graph = graph(
        sample_left_nodes(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, 2, 0b0001),
            (0, 1, 4, 0b0010),
            (1, 0, 6, 0b0100),
            (1, 1, 11, 0b1000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);

    assert!(
        bicliques
            .iter()
            .all(|biclique| biclique.left_node_ids != [0, 1] || biclique.right_node_ids != [0, 1])
    );
}

#[test]
fn test_enumerate_bicliques_rejects_overlapping_provenance() {
    let graph = graph(
        sample_left_nodes(),
        sample_right_nodes()[0..1].to_vec(),
        &[(0, 0, 2, 0b001), (1, 0, 6, 0b001)],
    );

    assert!(enumerate_bicliques(&graph).is_empty());
}

#[test]
fn test_enumerate_bicliques_emits_only_the_maximal_2x3_rectangle_once() {
    let graph = graph(
        sample_left_nodes(),
        sample_right_nodes(),
        &[
            (0, 0, 2, 0b000001),
            (0, 1, 4, 0b000010),
            (0, 2, 6, 0b000100),
            (1, 0, 6, 0b001000),
            (1, 1, 12, 0b010000),
            (1, 2, 18, 0b100000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);

    assert_eq!(bicliques.len(), 1);

    let biclique = &bicliques[0];
    assert_eq!(biclique.left_node_ids, vec![0, 1]);
    assert_eq!(biclique.right_node_ids, vec![0, 1, 2]);
    assert_eq!(
        biclique.left_coeffs,
        vec![Ratio::from_integer(1), Ratio::from_integer(3)]
    );
    assert_eq!(
        biclique.right_coeffs,
        vec![
            Ratio::from_integer(2),
            Ratio::from_integer(4),
            Ratio::from_integer(6)
        ]
    );
    assert_eq!(biclique.terms_used, 0b111111);
}

#[test]
fn test_enumerate_bicliques_ignores_non_current_left_pivots_after_bootstrap() {
    let graph = graph(
        sample_left_nodes(),
        sample_right_nodes()[0..2].to_vec(),
        &[
            (0, 0, 10, 0b001000),
            (0, 1, 20, 0b010000),
            (1, 0, 2, 0b000001),
            (1, 1, 4, 0b000010),
            (2, 0, 6, 0b000100),
            (2, 1, 12, 0b001000),
        ],
    );

    let bicliques = enumerate_bicliques(&graph);
    let biclique = find_biclique(&bicliques, &[1, 2], &[0, 1]);

    assert_eq!(
        biclique.left_coeffs,
        vec![Ratio::from_integer(1), Ratio::from_integer(3)]
    );
    assert_eq!(
        biclique.right_coeffs,
        vec![Ratio::from_integer(2), Ratio::from_integer(4)]
    );
    assert_eq!(biclique.terms_used, 0b001111);
}
