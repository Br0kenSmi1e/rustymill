use std::collections::HashMap;
use std::collections::HashSet;

use num::rational::Ratio;

use crate::canon::{canon_term, CanonTerm};
use crate::parenth::{FactorSubset, ParenthResult};
use crate::repr::{Index, IndexId, Rational, TensorComputation, TensorDef, Term};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VertexId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Left,
    Right,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LastStepIndices {
    pub left_ext: u64,
    pub right_ext: u64,
    pub sums: u64,
}

#[derive(Clone, Debug)]
pub struct EdgeInfo {
    pub term_idx: usize,
    pub eval_idx: usize,
    pub coeff: Rational,
    pub exc_cost: u64,
}

#[derive(Clone, Debug)]
pub struct ConstrGraph {
    pub vertices: Vec<CanonTerm>,
    pub vertex_side: Vec<Side>,
    pub edges: Vec<(VertexId, VertexId, EdgeInfo)>,
    pub last_step: LastStepIndices,
}

// ---------------------------------------------------------------------------
// Sub-term construction
// ---------------------------------------------------------------------------

/// Build a sub-Term from a subset of factors for canonicalization.
///
/// The sub-term contains only the factors indicated by the bitmask, with
/// sum_indices filtered to those that actually appear in the subset's factors.
/// The coefficient is always 1 (the real coefficient is tracked on the edge).
pub fn make_sub_term(term: &Term, subset: FactorSubset) -> Term {
    let mut factors = Vec::new();
    let mut s = subset;
    while s != 0 {
        let i = s.trailing_zeros() as usize;
        factors.push(term.factors[i].clone());
        s &= s - 1;
    }

    let mut present_ids: HashSet<IndexId> = HashSet::new();
    for factor in &factors {
        for &idx_id in &factor.indices {
            present_ids.insert(idx_id);
        }
    }

    let sum_indices: Vec<Index> = term
        .sum_indices
        .iter()
        .filter(|idx| present_ids.contains(&idx.id))
        .cloned()
        .collect();

    Term {
        coeff: Ratio::from_integer(1),
        sum_indices,
        factors,
    }
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

/// Build constriction graphs from a parenthesized tensor definition.
///
/// For each term with 2+ factors, every binary split (eval) of the full factor
/// set produces an edge in a bipartite graph.  The left and right sides of the
/// split are canonicalized to produce vertices.  Splits with the same
/// `LastStepIndices` (the external-index bitmasks of the two sides) are grouped
/// into a single `ConstrGraph`.
pub fn build_constr_graphs(
    def: &TensorDef,
    comp: &TensorComputation,
    parenth_results: &[ParenthResult],
) -> Vec<ConstrGraph> {
    // Accumulate edges grouped by LastStepIndices.
    // Each entry: (left_canon, right_canon, edge_info)
    let mut groups: HashMap<LastStepIndices, Vec<(CanonTerm, CanonTerm, EdgeInfo)>> =
        HashMap::new();

    for (term_idx, (term, pr)) in def.terms.iter().zip(parenth_results.iter()).enumerate() {
        if term.factors.len() < 2 {
            continue;
        }

        let n = pr.info.n_factors;
        let full_mask: FactorSubset = (1u64 << n) - 1;
        let interm = &pr.memoir[&full_mask];

        for (eval_idx, eval) in interm.evals.iter().enumerate() {
            let mut left_ext = pr.info.ext_bits(eval.left);
            let mut right_ext = pr.info.ext_bits(eval.right);
            let mut left_subset = eval.left;
            let mut right_subset = eval.right;

            // Normalize so left_ext <= right_ext.
            if left_ext > right_ext {
                std::mem::swap(&mut left_ext, &mut right_ext);
                std::mem::swap(&mut left_subset, &mut right_subset);
            }

            let lsi = LastStepIndices {
                left_ext,
                right_ext,
                sums: eval.contracted_sums,
            };

            let left_sub = make_sub_term(term, left_subset);
            let right_sub = make_sub_term(term, right_subset);

            let left_canon = canon_term(&left_sub, &def.ext_indices, comp.tensors());
            let right_canon = canon_term(&right_sub, &def.ext_indices, comp.tensors());

            let edge_info = EdgeInfo {
                term_idx,
                eval_idx,
                coeff: term.coeff.clone(),
                exc_cost: eval.cost - interm.best_cost,
            };

            groups
                .entry(lsi)
                .or_default()
                .push((left_canon, right_canon, edge_info));
        }
    }

    // Convert each group into a ConstrGraph.
    let mut result: Vec<ConstrGraph> = groups
        .into_iter()
        .map(|(lsi, entries)| {
            let mut vertex_map: HashMap<(CanonTerm, Side), VertexId> = HashMap::new();
            let mut vertices: Vec<CanonTerm> = Vec::new();
            let mut vertex_sides: Vec<Side> = Vec::new();
            let mut edges = Vec::new();

            for (left_canon, right_canon, edge_info) in entries {
                let left_vid = ensure_vertex(
                    &mut vertex_map,
                    &mut vertices,
                    &mut vertex_sides,
                    left_canon,
                    Side::Left,
                );
                let right_vid = ensure_vertex(
                    &mut vertex_map,
                    &mut vertices,
                    &mut vertex_sides,
                    right_canon,
                    Side::Right,
                );
                edges.push((left_vid, right_vid, edge_info));
            }

            ConstrGraph {
                vertices,
                vertex_side: vertex_sides,
                edges,
                last_step: lsi,
            }
        })
        .collect();

    // Sort for deterministic output.
    result.sort_by(|a, b| {
        a.last_step
            .left_ext
            .cmp(&b.last_step.left_ext)
            .then(a.last_step.right_ext.cmp(&b.last_step.right_ext))
    });

    result
}

/// Insert a vertex if it does not already exist, returning its id.
fn ensure_vertex(
    map: &mut HashMap<(CanonTerm, Side), VertexId>,
    verts: &mut Vec<CanonTerm>,
    sides: &mut Vec<Side>,
    canon: CanonTerm,
    side: Side,
) -> VertexId {
    let next_id = verts.len();
    *map.entry((canon.clone(), side)).or_insert_with(|| {
        verts.push(canon);
        sides.push(side);
        VertexId(next_id)
    })
}

// ---------------------------------------------------------------------------
// Cost coefficients
// ---------------------------------------------------------------------------

/// Precomputed cost coefficients for a constriction graph's index pattern.
#[derive(Clone, Debug)]
pub struct CostCoeffs {
    pub final_cost: u64,
    pub prep_left: u64,
    pub prep_right: u64,
}

/// Compute cost coefficients for a given index pattern.
/// Uses the parenth IndexInfo to look up index sizes.
pub fn compute_cost_coeffs(
    last_step: &LastStepIndices,
    info: &crate::parenth::IndexInfo,
) -> CostCoeffs {
    let left_ext_size = info.size_product_ext(last_step.left_ext).max(1);
    let right_ext_size = info.size_product_ext(last_step.right_ext).max(1);
    let sum_size = info.size_product_sum(last_step.sums).max(1);
    let ext_size = left_ext_size * right_ext_size;

    let contraction = if sum_size == 1 {
        ext_size
    } else {
        2 * ext_size * sum_size
    };
    let final_cost = contraction + ext_size;

    let prep_left = left_ext_size * sum_size;
    let prep_right = right_ext_size * sum_size;

    CostCoeffs {
        final_cost,
        prep_left,
        prep_right,
    }
}

/// Compute gross savings for adding a vertex to each side.
/// Returns (gross_for_adding_left, gross_for_adding_right).
pub fn gross_saving(coeffs: &CostCoeffs, n_left: usize, n_right: usize) -> (i64, i64) {
    if n_left == 0 || n_right == 0 {
        return (0, 0);
    }
    let gl = (n_right as i64) * (coeffs.final_cost as i64) - (coeffs.prep_left as i64);
    let gr = (n_left as i64) * (coeffs.final_cost as i64) - (coeffs.prep_right as i64);
    (gl, gr)
}
