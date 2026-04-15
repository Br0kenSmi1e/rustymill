use std::collections::HashMap;
use std::collections::HashSet;

use num::rational::Ratio;

use crate::canon::{canon_term, CanonTerm};
use crate::parenth::{FactorSubset, ParenthResult};
use crate::repr::{Factor, Index, IndexId, Range, RangeId, Rational, TensorComputation, TensorDef, TensorId, Term};

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

/// Grouping key for constriction graphs.
///
/// `left_ext` and `right_ext` are bitmasks of the definition's ext_indices
/// (globally consistent across terms).  `sums` is a sorted Vec of RangeIds
/// for the contracted sum indices (range-based, not term-local bitmask).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LastStepIndices {
    pub left_ext: u64,
    pub right_ext: u64,
    pub sums: Vec<RangeId>,
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

/// Compute the external indices of a sub-term: all indices appearing in its
/// factors that are not dummy (sum) indices within the sub-term.
fn sub_ext_indices(sub: &Term, full_term: &Term, def_ext: &[Index], comp: &TensorComputation) -> Vec<Index> {
    let dummy_ids: HashSet<IndexId> = sub.sum_indices.iter().map(|i| i.id).collect();
    // Build a map of all known IndexId -> RangeId from full_term, def_ext, and all comp definitions
    let mut all_indices: HashMap<IndexId, RangeId> = HashMap::new();
    for i in full_term.sum_indices.iter().chain(def_ext.iter()) {
        all_indices.insert(i.id, i.range);
    }
    for def in comp.definitions() {
        for i in &def.ext_indices {
            all_indices.insert(i.id, i.range);
        }
        for term in &def.terms {
            for i in &term.sum_indices {
                all_indices.insert(i.id, i.range);
            }
        }
    }
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for factor in &sub.factors {
        for &idx_id in &factor.indices {
            if !dummy_ids.contains(&idx_id) && seen.insert(idx_id) {
                if let Some(&range) = all_indices.get(&idx_id) {
                    result.push(Index { id: idx_id, range });
                }
            }
        }
    }
    result
}

/// Build constriction graphs from a parenthesized tensor definition.
///
/// For each term with 2+ factors, every binary split (eval) of the full factor
/// set produces an edge in a bipartite graph.  The left and right sides of the
/// split are canonicalized to produce vertices.  Splits with the same
/// `LastStepIndices` are grouped into a single `ConstrGraph`.
pub fn build_constr_graphs(
    def: &TensorDef,
    comp: &TensorComputation,
    parenth_results: &[ParenthResult],
) -> Vec<ConstrGraph> {
    // Accumulate edges grouped by LastStepIndices.
    let mut groups: HashMap<LastStepIndices, Vec<(CanonTerm, CanonTerm, EdgeInfo)>> =
        HashMap::new();

    for (term_idx, (term, pr)) in def.terms.iter().zip(parenth_results.iter()).enumerate() {
        if term.factors.len() < 2 {
            continue;
        }

        let n = pr.info.n_factors;
        let full_mask: FactorSubset = (1u64 << n) - 1;
        let interm = &pr.memoir[&(full_mask, 0)];

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

            // Convert contracted_sums bitmask to sorted Vec<RangeId>.
            let mut sum_ranges: Vec<RangeId> = Vec::new();
            let mut m = eval.contracted_sums;
            while m != 0 {
                let bit = m.trailing_zeros() as usize;
                sum_ranges.push(term.sum_indices[bit].range);
                m &= m - 1;
            }
            sum_ranges.sort();

            let lsi = LastStepIndices {
                left_ext,
                right_ext,
                sums: sum_ranges,
            };

            let left_sub = make_sub_term(term, left_subset);
            let right_sub = make_sub_term(term, right_subset);

            let left_canon = canon_term(&left_sub, &sub_ext_indices(&left_sub, term, &def.ext_indices, comp), comp.tensors());
            let right_canon = canon_term(&right_sub, &sub_ext_indices(&right_sub, term, &def.ext_indices, comp), comp.tensors());

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
            .then(a.last_step.sums.cmp(&b.last_step.sums))
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
    pub saving: u64,
    pub left: u64,
    pub right: u64,
}

/// Compute cost coefficients for a given index pattern.
pub fn compute_cost_coeffs(
    last_step: &LastStepIndices,
    info: &crate::parenth::IndexInfo,
    ranges: &[Range],
) -> CostCoeffs {
    let left_ext_size = info.size_product_ext(last_step.left_ext).max(1);
    let right_ext_size = info.size_product_ext(last_step.right_ext).max(1);
    let sum_size: u64 = last_step.sums.iter()
        .map(|r| ranges[r.0 as usize].size)
        .product::<u64>()
        .max(1);
    let ext_size = left_ext_size * right_ext_size;

    let contraction = if sum_size == 1 {
        ext_size
    } else {
        2 * ext_size * sum_size
    };

    let prep_left = left_ext_size * sum_size;
    let prep_right = right_ext_size * sum_size;

    CostCoeffs {
        saving: contraction + left_ext_size * right_ext_size,
        left: prep_left,
        right: prep_right,
    }
}

/// Compute gross savings for adding a vertex to each side.
/// Returns (gross_for_adding_left, gross_for_adding_right).
// pub fn gross_saving(coeffs: &CostCoeffs, n_left: usize, n_right: usize) -> (i64, i64) {
//     if n_left == 0 || n_right == 0 {
//         return (0, 0);
//     }
//     let gl = (n_right as i64) * (coeffs.final_cost as i64) - (coeffs.prep_left as i64);
//     let gr = (n_left as i64) * (coeffs.final_cost as i64) - (coeffs.prep_right as i64);
//     (gl, gr)
// }

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

    let mut biclique = Biclique {
        left_verts: Vec::new(),
        right_verts: Vec::new(),
        leading_coeff: None,
        terms_used: 0,
        saving: 0,
    };
    let mut cand: Vec<VertexId> = (0..n)
        .map(|v| VertexId(v))
        .collect();
    let mut subg: HashMap<VertexId, Delta> = (0..n)
        .map(|v| (VertexId(v), Delta::initial()))
        .collect();

    let mut results = Vec::new();

    expand(
        graph,
        coeffs,
        &mut biclique,
        &mut subg,
        &mut cand,
        &mut results,
    );

    results
}

pub fn expand(
    graph: &ConstrGraph,
    coeffs: &CostCoeffs,
    biclique: &mut Biclique,
    subg: &mut HashMap<VertexId, Delta>,
    cand: &mut Vec<VertexId>,
    results: &mut Vec<Biclique>,
) {
    // update subg saving
    let mut is_maximal = false;
    for (q, delta) in subg.iter_mut() {
        let q_side = graph.vertex_side[q.0];
        delta.saving = match q_side {
            Side::Left => - (coeffs.left as i64) + (biclique.right_verts.len() as i64) * (coeffs.saving as i64),
            Side::Right => - (coeffs.right as i64) + (biclique.left_verts.len() as i64) * (coeffs.saving as i64),
        } - delta.exc_cost;
        is_maximal |= delta.saving > 0;
    }
    // verify maximal & profitable, return
    let has_sharing = biclique.left_verts.len() >= 2 || biclique.right_verts.len() >= 2;
    if !is_maximal && has_sharing && (biclique.saving > 0) {
        results.push(biclique.clone());
    }
    // quadratic loop to prune cand
    let mut subgq: HashMap<VertexId, HashMap<VertexId, Delta>> = HashMap::new();
    for (q, dq) in subg.iter() {
        for (r, dr) in subg.iter() {
            if let Some(update) = update_delta(graph, biclique, *q, dq, *r, dr) {
                subgq
                    .entry(*q)
                    .or_insert_with(HashMap::new)
                    .insert(*r, update);
            }
        }
    }
    // add cand and recurse
    let curr: Vec<VertexId> = sift(graph, biclique, cand, subg, &subgq);
    for i in 0..curr.len() {
        let q = curr[i];
        if let Some(dq) = subg.get(&q) {
            if let Some(idx) = cand.iter().position(|v| *v == q) {
                cand.remove(idx);
            }
            push(biclique, q, graph.vertex_side[q.0], dq);
            let mut empty = HashMap::new();
            let sub = subgq.get_mut(&q).unwrap_or(&mut empty);
            expand(graph, coeffs, biclique, sub, cand, results);
            pop(biclique, q, graph.vertex_side[q.0], dq);
        }
    }
}

pub fn update_delta(
    graph: &ConstrGraph,
    biclique: &Biclique,
    q: VertexId,
    dq: &Delta,
    r: VertexId,
    dr: &Delta,
) -> Option<Delta> {
    let sq = graph.vertex_side[q.0];
    let sr = graph.vertex_side[r.0];

    if (dq.terms & dr.terms) != 0 {
        return None;
    }

    if sq == sr {
        if let Some(dq_lc) = &dq.leading_coeff {
            if let Some(dr_lc) = &dr.leading_coeff {
                let mut new_dr = dr.clone();
                new_dr.leading_coeff = None;
                new_dr.coeff = dr_lc / dq_lc;
                Some(new_dr)
            } else {
                None
            }
        } else {
            Some(dr.clone())
        }
    } else {
        let edges = graph.edges_between(q, r);
        if edges.is_empty() {
            return None;
        }

        let bitmask: u64 = edges.iter()
            .filter(|e| e.term_idx < 64)
            .map(|e| 1u64 << e.term_idx)
            .fold(0u64, |acc, mask| acc | mask);
        
        if (bitmask & dq.terms) != 0
            || (bitmask & biclique.terms_used) != 0
        {
            return None;
        }

        let mut new_dr = dr.clone();
        new_dr.terms |= bitmask;
        
        let first_edge = &edges[0];
        new_dr.exc_cost += first_edge.exc_cost as i64;

        let total_coeff: Rational = edges.iter()
            .map(|e| e.coeff.clone())
            .sum();

        if let Some(dq_lc) = &dq.leading_coeff {
            new_dr.coeff = &total_coeff / dq_lc;
        } else if biclique.leading_coeff.is_none() {
            new_dr.leading_coeff = Some(total_coeff);
        } else {
            if let Some(bic_lc) = &biclique.leading_coeff {
                let expected = bic_lc * &dq.coeff * &dr.coeff;
                if total_coeff != expected {
                    return None;
                }
            }
        }
                
        Some(new_dr)
    }
}

pub fn sift(
    graph: &ConstrGraph,
    biclique: &Biclique,
    cand: &[VertexId],
    subg: &HashMap<VertexId, Delta>,
    subgq: &HashMap<VertexId, HashMap<VertexId, Delta>>,
) -> Vec<VertexId> {
    if biclique.left_verts.is_empty() {
        let r: Vec<VertexId> = cand.iter()
            .filter(|q| graph.vertex_side[q.0] == Side::Left)
            .copied()
            .collect();
        return r;
    }

    let curr: Vec<VertexId> = if biclique.right_verts.is_empty() {
        cand.iter()
            .filter(|q| graph.vertex_side[q.0] == Side::Right)
            .copied()
            .collect()
    } else {
        cand.iter()
            .filter(|q| subg.get(q).map_or(false, |delta| delta.saving > 0))
            .copied()
            .collect()
    };
    let mut best_f: Vec<VertexId> = Vec::new();
    let mut max_intersection = 0;

    for &u in subg.keys() {
        let u_side = graph.vertex_side[u.0];

        if let Some(neighbors) = subgq.get(&u) {
            let f_u: Vec<VertexId> = neighbors
                .keys()
                .filter(|&v| graph.vertex_side[v.0] == u_side)
                .copied()
                .collect();

            let intersection_size = f_u.iter().filter(|v| curr.contains(v)).count();

            if intersection_size > max_intersection {
                max_intersection = intersection_size;
                best_f = f_u;
            }
        }
    }

    let result: Vec<VertexId> = curr.into_iter()
        .filter(|v| !best_f.contains(v))
        .collect();
    result
}

pub fn push(
    biclique: &mut Biclique,
    q: VertexId,
    side: Side,
    dq: &Delta,
) {
    biclique.saving += dq.saving;
    biclique.terms_used |= dq.terms;
    match side {
        Side::Left => biclique.left_verts.push((q, dq.coeff.clone())),
        Side::Right => biclique.right_verts.push((q, dq.coeff.clone())),
    };
    if let Some(dq_lc) = &dq.leading_coeff {
        biclique.leading_coeff = Some(dq_lc.clone());
    }
}

pub fn pop(
    biclique: &mut Biclique,
    q: VertexId,
    side: Side,
    dq: &Delta,
) {
    biclique.saving -= dq.saving;
    biclique.terms_used ^= dq.terms;
    match side {
        Side::Left => biclique.left_verts.pop(),
        Side::Right => biclique.right_verts.pop(),
    };
}

// ---------------------------------------------------------------------------
// Factorization conversion
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Factorization {
    pub terms_consumed: Vec<usize>,
    pub intermediates: Vec<TensorDef>,
    pub replacement_term: Term,
    pub saving: i64,
}

/// Convert bicliques into concrete `Factorization` records.
pub fn factorizations(
    def: &TensorDef,
    parenth_results: &[ParenthResult],
    comp: &TensorComputation,
    next_tensor_id: TensorId,
) -> Vec<Factorization> {
    let graphs = build_constr_graphs(def, comp, parenth_results);
    let mut results = Vec::new();

    let mut seen_vertex_sets: HashSet<(Vec<VertexId>, Vec<VertexId>)> = HashSet::new();

    for graph in &graphs {
        if graph.edges.is_empty() {
            continue;
        }

        let first_term_idx = graph.edges[0].2.term_idx;
        let info = &parenth_results[first_term_idx].info;
        let coeffs = compute_cost_coeffs(&graph.last_step, info, comp.ranges());
        let bicliques = find_bicliques(graph, &coeffs);

        for bc in bicliques {
            if bc.saving <= 0 {
                continue;
            }

            let is_complete = bc.left_verts.iter().all(|(lv, _)| {
                bc.right_verts
                    .iter()
                    .all(|(rv, _)| !graph.edges_between(*lv, *rv).is_empty())
            });
            if !is_complete {
                continue;
            }

            let mut left_ids: Vec<VertexId> = bc.left_verts.iter().map(|(v, _)| *v).collect();
            let mut right_ids: Vec<VertexId> = bc.right_verts.iter().map(|(v, _)| *v).collect();
            left_ids.sort();
            right_ids.sort();
            if !seen_vertex_sets.insert((left_ids, right_ids)) {
                continue;
            }

            let terms_consumed = bits_to_vec(bc.terms_used);

            // Use the first consumed term as reference for index naming.
            let repr_term = &def.terms[terms_consumed[0]];

            // Reconstruct contracted sum Indices from the LastStepIndices ranges
            // matched against repr_term's sum_indices.
            let contracted_sums = match_contracted_sums(
                &graph.last_step.sums, repr_term,
            );
            let contracted_ids: HashSet<IndexId> =
                contracted_sums.iter().map(|i| i.id).collect();

            let left_ext = bits_to_indices(graph.last_step.left_ext, &def.ext_indices);
            let right_ext = bits_to_indices(graph.last_step.right_ext, &def.ext_indices);

            let ext_ids: HashSet<IndexId> = def.ext_indices.iter().map(|i| i.id).collect();

            let mut intermediates = Vec::new();

            let mut candidate_id = next_tensor_id.0;
            let (left_tid, left_indices) = build_side_ref(
                &bc.left_verts, Side::Left, graph, def, parenth_results,
                &left_ext, &contracted_sums, &contracted_ids, &ext_ids,
                repr_term, &mut intermediates, &mut candidate_id, comp,
            );

            let (right_tid, right_indices) = build_side_ref(
                &bc.right_verts, Side::Right, graph, def, parenth_results,
                &right_ext, &contracted_sums, &contracted_ids, &ext_ids,
                repr_term, &mut intermediates, &mut candidate_id, comp,
            );

            let coeff = bc
                .leading_coeff
                .unwrap_or_else(|| Ratio::from_integer(1));

            let replacement_term = Term {
                coeff,
                sum_indices: contracted_sums,
                factors: vec![
                    Factor { tensor: left_tid, indices: left_indices },
                    Factor { tensor: right_tid, indices: right_indices },
                ],
            };

            results.push(Factorization {
                terms_consumed,
                intermediates,
                replacement_term,
                saving: bc.saving,
            });
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Factorization helpers
// ---------------------------------------------------------------------------

/// Expand a bitmask into a sorted Vec of set-bit positions.
fn bits_to_vec(mut mask: u64) -> Vec<usize> {
    let mut out = Vec::new();
    while mask != 0 {
        out.push(mask.trailing_zeros() as usize);
        mask &= mask - 1;
    }
    out
}

/// Map set bits to the corresponding elements of `source`.
fn bits_to_indices(mut mask: u64, source: &[Index]) -> Vec<Index> {
    let mut out = Vec::new();
    while mask != 0 {
        let bit = mask.trailing_zeros() as usize;
        out.push(source[bit].clone());
        mask &= mask - 1;
    }
    out
}

/// Reconstruct contracted sum Index objects from a sorted list of RangeIds,
/// matched against the repr_term's sum_indices by range.
fn match_contracted_sums(
    sum_ranges: &[RangeId],
    repr_term: &Term,
) -> Vec<Index> {
    let mut used: HashSet<usize> = HashSet::new();
    let mut result = Vec::new();
    for &range in sum_ranges {
        for (i, idx) in repr_term.sum_indices.iter().enumerate() {
            if !used.contains(&i) && idx.range == range {
                result.push(idx.clone());
                used.insert(i);
                break;
            }
        }
    }
    result
}

/// Build a remap from a vertex's term's sum IndexIds to the repr_term's IndexIds.
/// Matches contracted sums by range.
fn build_contracted_remap(
    vertex_term: &Term,
    repr_term: &Term,
    contracted_ids: &HashSet<IndexId>,
    ext_ids: &HashSet<IndexId>,
) -> HashMap<IndexId, IndexId> {
    let mut remap = HashMap::new();
    let mut used: HashSet<usize> = HashSet::new();

    // For each sum index in the vertex's term that is a contracted sum,
    // find the corresponding repr_term sum index by range.
    for v_idx in &vertex_term.sum_indices {
        if ext_ids.contains(&v_idx.id) {
            continue;
        }
        // Find matching repr_term sum index by range
        for (ri, r_idx) in repr_term.sum_indices.iter().enumerate() {
            if !used.contains(&ri) && r_idx.range == v_idx.range && contracted_ids.contains(&r_idx.id) {
                if v_idx.id != r_idx.id {
                    remap.insert(v_idx.id, r_idx.id);
                }
                used.insert(ri);
                break;
            }
        }
    }

    remap
}

/// Apply an IndexId remap to a sub-term's factors and sum_indices.
fn apply_remap(sub: &mut Term, remap: &HashMap<IndexId, IndexId>) {
    if remap.is_empty() {
        return;
    }
    for factor in &mut sub.factors {
        for idx in &mut factor.indices {
            if let Some(&new_id) = remap.get(idx) {
                *idx = new_id;
            }
        }
    }
    for idx in &mut sub.sum_indices {
        if let Some(&new_id) = remap.get(&idx.id) {
            idx.id = new_id;
        }
    }
}

/// For a vertex on a given side, find the factor subset in the original term.
fn vertex_subset(
    v: VertexId,
    side: Side,
    graph: &ConstrGraph,
    parenth_results: &[ParenthResult],
) -> (usize, FactorSubset) {
    let (_, _, ei) = match side {
        Side::Left => graph
            .edges
            .iter()
            .find(|(l, _, _)| *l == v)
            .expect("vertex must have an edge"),
        Side::Right => graph
            .edges
            .iter()
            .find(|(_, r, _)| *r == v)
            .expect("vertex must have an edge"),
    };

    let pr = &parenth_results[ei.term_idx];
    let n = pr.info.n_factors;
    let full_mask: FactorSubset = (1u64 << n) - 1;
    let interm = &pr.memoir[&(full_mask, 0)];
    let eval = &interm.evals[ei.eval_idx];

    let left_ext = pr.info.ext_bits(eval.left);
    let right_ext = pr.info.ext_bits(eval.right);
    let (mut ls, mut rs) = (eval.left, eval.right);
    if left_ext > right_ext {
        std::mem::swap(&mut ls, &mut rs);
    }

    let subset = match side {
        Side::Left => ls,
        Side::Right => rs,
    };

    (ei.term_idx, subset)
}

/// Build either a new intermediate TensorDef (when >1 vertex) or a direct
/// tensor reference (single-factor single vertex).
#[allow(clippy::too_many_arguments)]
fn build_side_ref(
    verts: &[(VertexId, Rational)],
    side: Side,
    graph: &ConstrGraph,
    def: &TensorDef,
    parenth_results: &[ParenthResult],
    side_ext: &[Index],
    contracted_sums: &[Index],
    contracted_ids: &HashSet<IndexId>,
    ext_ids: &HashSet<IndexId>,
    repr_term: &Term,
    intermediates: &mut Vec<TensorDef>,
    next_id: &mut u32,
    _comp: &TensorComputation,
) -> (TensorId, Vec<IndexId>) {
    let interm_ext: Vec<Index> = side_ext
        .iter()
        .chain(contracted_sums.iter())
        .cloned()
        .collect();
    let interm_idx_ids: Vec<IndexId> = interm_ext.iter().map(|i| i.id).collect();

    if verts.len() > 1 {
        let tid = TensorId(*next_id);
        *next_id += 1;

        let mut terms = Vec::new();
        for &(v, ref coeff) in verts {
            let (term_idx, subset) = vertex_subset(v, side, graph, parenth_results);
            let mut sub = make_sub_term(&def.terms[term_idx], subset);

            // Bug 5 fix: remap dummy indices to match repr_term's naming.
            let remap = build_contracted_remap(
                &def.terms[term_idx], repr_term, contracted_ids, ext_ids,
            );
            apply_remap(&mut sub, &remap);

            let sum_indices: Vec<Index> = sub
                .sum_indices
                .iter()
                .filter(|idx| !contracted_ids.contains(&idx.id))
                .cloned()
                .collect();

            terms.push(Term {
                coeff: coeff.clone(),
                sum_indices,
                factors: sub.factors,
            });
        }

        intermediates.push(TensorDef {
            base: tid,
            ext_indices: interm_ext,
            terms,
        });

        (tid, interm_idx_ids)
    } else {
        let (term_idx, subset) = vertex_subset(verts[0].0, side, graph, parenth_results);
        let mut sub = make_sub_term(&def.terms[term_idx], subset);

        // Bug 5 fix: remap for single vertex too.
        let remap = build_contracted_remap(
            &def.terms[term_idx], repr_term, contracted_ids, ext_ids,
        );
        apply_remap(&mut sub, &remap);

        if sub.factors.len() == 1 {
            let f = &sub.factors[0];
            (f.tensor, f.indices.clone())
        } else {
            let tid = TensorId(*next_id);
            *next_id += 1;

            let sum_indices: Vec<Index> = sub
                .sum_indices
                .iter()
                .filter(|idx| !contracted_ids.contains(&idx.id))
                .cloned()
                .collect();

            intermediates.push(TensorDef {
                base: tid,
                ext_indices: interm_ext,
                terms: vec![Term {
                    coeff: verts[0].1.clone(),
                    sum_indices,
                    factors: sub.factors,
                }],
            });

            (tid, interm_idx_ids)
        }
    }
}
