use num::rational::Ratio;

use rustymill::constr::{build_constr_graphs, Side};
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
