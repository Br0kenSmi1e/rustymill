use num::rational::Ratio;

use rustymill::constr::{
    build_constr_graphs, compute_cost_coeffs, gross_saving, update_delta, BronKerboschState, Delta,
    Side,
};
use rustymill::parenth::{parenthesize, ParenthResult};
use rustymill::repr::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build: t[a,b] = X[a,c]*Z[c,b] + Y[a,c]*Z[c,b]
///
/// All ranges have size 10 (occ).  Terms share the right factor Z[c,b].
fn make_shared_factor_def() -> (TensorComputation, TensorDef, Vec<ParenthResult>) {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);

    // Tensors: t(0), X(1), Y(2), Z(3)
    let _t = comp.add_tensor(&[occ, occ], vec![]);
    let _x = comp.add_tensor(&[occ, occ], vec![]);
    let _y = comp.add_tensor(&[occ, occ], vec![]);
    let _z = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);

    let ext_indices = vec![
        Index { id: a, range: occ },
        Index { id: b, range: occ },
    ];

    let term0 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: c, range: occ }],
        factors: vec![
            Factor {
                tensor: TensorId(1),
                indices: vec![a, c],
            },
            Factor {
                tensor: TensorId(3),
                indices: vec![c, b],
            },
        ],
    };

    let term1 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: c, range: occ }],
        factors: vec![
            Factor {
                tensor: TensorId(2),
                indices: vec![a, c],
            },
            Factor {
                tensor: TensorId(3),
                indices: vec![c, b],
            },
        ],
    };

    let def = TensorDef {
        base: TensorId(0),
        ext_indices: ext_indices.clone(),
        terms: vec![term0.clone(), term1.clone()],
    };

    let pr0 = parenthesize(&term0, &ext_indices, comp.ranges());
    let pr1 = parenthesize(&term1, &ext_indices, comp.ranges());

    (comp, def, vec![pr0, pr1])
}

/// Build: t[a,b] = X[a,c]*Z[c,b] + Y[a,c]*Z[c,b] + W[a,b]
///
/// W has only 1 factor and should be excluded from the graph.
fn make_mixed_terms_def() -> (TensorComputation, TensorDef, Vec<ParenthResult>) {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);

    let _t = comp.add_tensor(&[occ, occ], vec![]);
    let _x = comp.add_tensor(&[occ, occ], vec![]);
    let _y = comp.add_tensor(&[occ, occ], vec![]);
    let _z = comp.add_tensor(&[occ, occ], vec![]);
    let _w = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);

    let ext_indices = vec![
        Index { id: a, range: occ },
        Index { id: b, range: occ },
    ];

    let term0 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: c, range: occ }],
        factors: vec![
            Factor {
                tensor: TensorId(1),
                indices: vec![a, c],
            },
            Factor {
                tensor: TensorId(3),
                indices: vec![c, b],
            },
        ],
    };

    let term1 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: c, range: occ }],
        factors: vec![
            Factor {
                tensor: TensorId(2),
                indices: vec![a, c],
            },
            Factor {
                tensor: TensorId(3),
                indices: vec![c, b],
            },
        ],
    };

    // W[a,b] — single factor, no summation indices.
    let term2 = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![],
        factors: vec![Factor {
            tensor: TensorId(4),
            indices: vec![a, b],
        }],
    };

    let def = TensorDef {
        base: TensorId(0),
        ext_indices: ext_indices.clone(),
        terms: vec![term0.clone(), term1.clone(), term2.clone()],
    };

    let pr0 = parenthesize(&term0, &ext_indices, comp.ranges());
    let pr1 = parenthesize(&term1, &ext_indices, comp.ranges());
    let pr2 = parenthesize(&term2, &ext_indices, comp.ranges());

    (comp, def, vec![pr0, pr1, pr2])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_build_constr_graphs_shared_factor() {
    let (comp, def, prs) = make_shared_factor_def();
    let graphs = build_constr_graphs(&def, &comp, &prs);

    // One constriction graph (all evals share the same LastStepIndices).
    assert_eq!(graphs.len(), 1);
    let g = &graphs[0];

    // 3 vertices: X (left), Y (left), Z (right).
    assert_eq!(g.vertices.len(), 3);

    // 2 edges: (X, Z) from term 0, (Y, Z) from term 1.
    assert_eq!(g.edges.len(), 2);

    // Verify bipartite structure: each edge connects a Left to a Right vertex.
    for (l, r, _) in &g.edges {
        assert_eq!(g.vertex_side[l.0], Side::Left);
        assert_eq!(g.vertex_side[r.0], Side::Right);
    }

    // The right vertex should be shared (same VertexId) across both edges.
    assert_eq!(g.edges[0].1, g.edges[1].1);

    // The left vertices should be distinct.
    assert_ne!(g.edges[0].0, g.edges[1].0);
}

#[test]
fn test_build_constr_graphs_single_factor_term_excluded() {
    let (comp, def, prs) = make_mixed_terms_def();
    let graphs = build_constr_graphs(&def, &comp, &prs);

    // Still 1 graph — W is excluded because it has only 1 factor.
    assert_eq!(graphs.len(), 1);
    let g = &graphs[0];

    // 3 vertices (X, Y on left; Z on right). W absent.
    assert_eq!(g.vertices.len(), 3);

    // 2 edges, same as the shared-factor case.
    assert_eq!(g.edges.len(), 2);
}

#[test]
fn test_cost_coeffs() {
    let (comp, def, prs) = make_shared_factor_def();
    let graphs = build_constr_graphs(&def, &comp, &prs);
    let graph = &graphs[0];

    let coeffs = compute_cost_coeffs(&graph.last_step, &prs[0].info);
    // All occ=10. left_ext={a}=10, right_ext={b}=10, sums={c}=10
    // ext_size = 10*10 = 100, sum_size = 10
    // final_cost = 2*100*10 + 100 = 2100
    // prep_left = 10*10 = 100
    // prep_right = 10*10 = 100
    assert_eq!(coeffs.final_cost, 2100);
    assert_eq!(coeffs.prep_left, 100);
    assert_eq!(coeffs.prep_right, 100);
}

#[test]
fn test_gross_saving() {
    let (comp, def, prs) = make_shared_factor_def();
    let graphs = build_constr_graphs(&def, &comp, &prs);
    let graph = &graphs[0];
    let coeffs = compute_cost_coeffs(&graph.last_step, &prs[0].info);

    let (gl, gr) = gross_saving(&coeffs, 1, 1);
    // gl = 1 * 2100 - 100 = 2000
    // gr = 1 * 2100 - 100 = 2000
    assert_eq!(gl, 2000);
    assert_eq!(gr, 2000);

    let (gl, gr) = gross_saving(&coeffs, 2, 1);
    // gl = 1 * 2100 - 100 = 2000
    // gr = 2 * 2100 - 100 = 4100
    assert_eq!(gl, 2000);
    assert_eq!(gr, 4100);
}

#[test]
fn test_delta_different_parts_first_edge() {
    let (comp, def, prs) = make_shared_factor_def();
    let graphs = build_constr_graphs(&def, &comp, &prs);
    let graph = &graphs[0];
    let coeffs = compute_cost_coeffs(&graph.last_step, &prs[0].info);

    let left_verts = graph.vertices_on_side(Side::Left);
    let right_verts = graph.vertices_on_side(Side::Right);
    assert!(!left_verts.is_empty());
    assert!(!right_verts.is_empty());

    let left_v = left_verts[0];
    let right_v = right_verts[0];
    let initial = Delta::initial();
    let bk = BronKerboschState::new();

    let result = update_delta(graph, &coeffs, &bk, left_v, &initial, right_v, &initial);
    assert!(result.is_some());
    let delta = result.unwrap();
    // First cross-part edge sets leading_coeff
    assert!(delta.leading_coeff.is_some());
}

#[test]
fn test_delta_same_part_no_constraint() {
    let (comp, def, prs) = make_shared_factor_def();
    let graphs = build_constr_graphs(&def, &comp, &prs);
    let graph = &graphs[0];
    let coeffs = compute_cost_coeffs(&graph.last_step, &prs[0].info);

    let left_verts = graph.vertices_on_side(Side::Left);
    if left_verts.len() >= 2 {
        let v0 = left_verts[0];
        let v1 = left_verts[1];
        let d0 = Delta::initial();
        let d1 = Delta::initial();
        let bk = BronKerboschState::new();

        let result = update_delta(graph, &coeffs, &bk, v0, &d0, v1, &d1);
        assert!(result.is_some());
    }
}
