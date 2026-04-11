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

// ---------------------------------------------------------------------------
// ConstrGraph helpers
// ---------------------------------------------------------------------------

impl ConstrGraph {
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn vertices_on_side(&self, side: Side) -> Vec<VertexId> {
        self.vertex_side
            .iter()
            .enumerate()
            .filter(|(_, &s)| s == side)
            .map(|(i, _)| VertexId(i))
            .collect()
    }

    /// Get all edges between two vertices.
    pub fn edges_between(&self, u: VertexId, v: VertexId) -> Vec<&EdgeInfo> {
        self.edges
            .iter()
            .filter(|(a, b, _)| (*a == u && *b == v) || (*a == v && *b == u))
            .map(|(_, _, info)| info)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Delta computation for Bron-Kerbosch biclique search
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Delta {
    pub coeff: Rational,
    pub leading_coeff: Option<Rational>,
    pub terms: u64,
    pub exc_cost: i64,
    pub saving: i64,
}

impl Delta {
    pub fn initial() -> Self {
        Delta {
            coeff: Ratio::from_integer(1),
            leading_coeff: None,
            terms: 0,
            exc_cost: 0,
            saving: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BronKerboschState {
    pub leading_coeff: Option<Rational>,
    pub terms_used: u64,
    pub n_left: usize,
    pub n_right: usize,
}

impl BronKerboschState {
    pub fn new() -> Self {
        BronKerboschState {
            leading_coeff: None,
            terms_used: 0,
            n_left: 0,
            n_right: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Biclique enumeration (Bron-Kerbosch variant)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Biclique {
    pub left_verts: Vec<(VertexId, Rational)>,
    pub right_verts: Vec<(VertexId, Rational)>,
    pub leading_coeff: Option<Rational>,
    pub terms_used: u64,
    pub saving: i64,
}

pub fn find_bicliques(graph: &ConstrGraph, coeffs: &CostCoeffs) -> Vec<Biclique> {
    let n = graph.num_vertices();
    if n == 0 {
        return Vec::new();
    }

    let subg: Vec<(VertexId, Delta)> = (0..n)
        .map(|v| (VertexId(v), Delta::initial()))
        .collect();

    let mut results = Vec::new();
    let mut state = BronKerboschState::new();

    expand(
        graph,
        coeffs,
        &mut state,
        &mut Vec::new(),
        &mut Vec::new(),
        subg,
        &mut results,
    );

    results
}

fn expand(
    graph: &ConstrGraph,
    coeffs: &CostCoeffs,
    state: &mut BronKerboschState,
    left_verts: &mut Vec<(VertexId, Rational)>,
    right_verts: &mut Vec<(VertexId, Rational)>,
    subg: Vec<(VertexId, Delta)>,
    results: &mut Vec<Biclique>,
) {
    let n_left = left_verts.len();
    let n_right = right_verts.len();

    // Check maximality: all candidates have negative saving
    let is_maximal = subg.iter().all(|(_, d)| d.saving < 0);
    let is_profitable = n_left > 0 && n_right > 0 && (n_left > 1 || n_right > 1);

    if is_maximal && is_profitable {
        // Compute total exc_cost from edges in the biclique
        let mut total_exc_cost: i64 = 0;
        let mut terms_used = 0u64;
        for &(lv, _) in left_verts.iter() {
            for &(rv, _) in right_verts.iter() {
                let edges = graph.edges_between(lv, rv);
                for edge in &edges {
                    let et = 1u64 << edge.term_idx;
                    if terms_used & et == 0 {
                        total_exc_cost += edge.exc_cost as i64;
                        terms_used |= et;
                        break;
                    }
                }
            }
        }

        let saving = (n_left * n_right) as i64 * coeffs.final_cost as i64
            - coeffs.final_cost as i64
            - (n_left.saturating_sub(1)) as i64 * coeffs.prep_left as i64
            - (n_right.saturating_sub(1)) as i64 * coeffs.prep_right as i64
            - total_exc_cost;

        if saving >= 0 {
            results.push(Biclique {
                left_verts: left_verts.clone(),
                right_verts: right_verts.clone(),
                leading_coeff: state.leading_coeff.clone(),
                terms_used,
                saving,
            });
        }
    }

    // Collect candidates with non-negative saving
    let candidates: Vec<(VertexId, Delta)> = subg
        .iter()
        .filter(|(_, d)| d.saving >= 0)
        .cloned()
        .collect();

    for (q_v, q_d) in &candidates {
        let q_v = *q_v;
        let q_side = graph.vertex_side[q_v.0];

        // Compute updated subgraph
        let mut new_subg = Vec::new();
        for &(s_v, ref s_d) in &subg {
            if s_v == q_v {
                continue;
            }
            match update_delta(graph, coeffs, state, q_v, q_d, s_v, s_d) {
                Some(mut updated) => {
                    let mut nl = state.n_left;
                    let mut nr = state.n_right;
                    match q_side {
                        Side::Left => nl += 1,
                        Side::Right => nr += 1,
                    }
                    let (gl, gr) = gross_saving(coeffs, nl, nr);
                    let gross = match graph.vertex_side[s_v.0] {
                        Side::Left => gl,
                        Side::Right => gr,
                    };
                    updated.saving = gross - updated.exc_cost;
                    new_subg.push((s_v, updated));
                }
                None => {
                    let mut excluded = s_d.clone();
                    excluded.saving = -1;
                    new_subg.push((s_v, excluded));
                }
            }
        }

        // Save state
        let prev_leading = state.leading_coeff.clone();
        let prev_terms = state.terms_used;
        let prev_n_left = state.n_left;
        let prev_n_right = state.n_right;

        // Add vertex
        match q_side {
            Side::Left => {
                left_verts.push((q_v, q_d.coeff.clone()));
                state.n_left += 1;
            }
            Side::Right => {
                right_verts.push((q_v, q_d.coeff.clone()));
                state.n_right += 1;
            }
        }
        if let Some(ref lc) = q_d.leading_coeff {
            state.leading_coeff = Some(lc.clone());
        }
        state.terms_used |= q_d.terms;

        expand(graph, coeffs, state, left_verts, right_verts, new_subg, results);

        // Backtrack
        match q_side {
            Side::Left => {
                left_verts.pop();
            }
            Side::Right => {
                right_verts.pop();
            }
        }
        state.leading_coeff = prev_leading;
        state.terms_used = prev_terms;
        state.n_left = prev_n_left;
        state.n_right = prev_n_right;
    }
}

/// Update a delta for `curr_v` when considering adding `new_v` to the biclique.
///
/// Returns `None` if adding `new_v` is incompatible with `curr_v`.
pub fn update_delta(
    graph: &ConstrGraph,
    _coeffs: &CostCoeffs,
    bk_state: &BronKerboschState,
    new_v: VertexId,
    new_d: &Delta,
    curr_v: VertexId,
    curr_d: &Delta,
) -> Option<Delta> {
    let new_side = graph.vertex_side[new_v.0];
    let curr_side = graph.vertex_side[curr_v.0];

    let mut updated = Delta {
        coeff: curr_d.coeff.clone(),
        leading_coeff: curr_d.leading_coeff.clone(),
        terms: curr_d.terms,
        exc_cost: curr_d.exc_cost,
        saving: 0, // computed later by caller
    };

    if new_side == curr_side {
        // Same part
        if new_d.leading_coeff.is_some() && curr_d.leading_coeff.is_some() {
            let new_lc = new_d.leading_coeff.as_ref().unwrap();
            let curr_lc = curr_d.leading_coeff.as_ref().unwrap();
            if *new_lc == Ratio::from_integer(0) {
                return None;
            }
            updated.coeff = curr_lc / new_lc;
            updated.leading_coeff = None;
        }
    } else {
        // Different parts: must have edge
        let edges = graph.edges_between(new_v, curr_v);
        if edges.is_empty() {
            return None;
        }

        // Pick best compatible edge (lowest exc_cost, no term conflict)
        let mut best_edge: Option<&EdgeInfo> = None;
        for edge in &edges {
            let edge_term = 1u64 << edge.term_idx;
            let conflict = (edge_term & new_d.terms != 0)
                || (edge_term & curr_d.terms != 0)
                || (edge_term & bk_state.terms_used != 0);
            if conflict {
                continue;
            }
            match best_edge {
                None => best_edge = Some(edge),
                Some(prev) if edge.exc_cost < prev.exc_cost => best_edge = Some(edge),
                _ => {}
            }
        }

        let edge = best_edge?;

        let edge_term = 1u64 << edge.term_idx;
        updated.terms |= edge_term;
        updated.exc_cost += edge.exc_cost as i64;

        let edge_coeff = &edge.coeff;

        if let Some(ref new_lc) = new_d.leading_coeff {
            if *new_lc == Ratio::from_integer(0) {
                return None;
            }
            updated.coeff = edge_coeff / new_lc;
        } else if bk_state.leading_coeff.is_none() {
            updated.leading_coeff = Some(edge_coeff.clone());
        } else {
            let lc = bk_state.leading_coeff.as_ref().unwrap();
            let expected = lc * &new_d.coeff * &curr_d.coeff;
            if *edge_coeff != expected {
                return None;
            }
        }
    }

    Some(updated)
}
