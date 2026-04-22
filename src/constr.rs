use std::collections::HashMap;
use std::collections::HashSet;

use num::rational::Ratio;

use crate::canon::{build_canon_pool, canonicalize_sub_term};
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
    pub vertices: Vec<Term>,
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
pub fn build_constr_graphs(
    def: &TensorDef,
    comp: &TensorComputation,
    parenth_results: &[ParenthResult],
) -> Vec<ConstrGraph> {
    let pool = build_canon_pool(def);

    let ext_ids: HashSet<IndexId> = def.ext_indices.iter().map(|i| i.id).collect();
    let ext_range: HashMap<IndexId, RangeId> = def.ext_indices.iter().map(|i| (i.id, i.range)).collect();

    let mut groups_left: HashMap<LastStepIndices, Vec<(Term, Term, EdgeInfo)>> = HashMap::new();
    let mut groups_right: HashMap<LastStepIndices, Vec<(Term, Term, EdgeInfo)>> = HashMap::new();

    for (term_idx, (term, pr)) in def.terms.iter().zip(parenth_results.iter()).enumerate() {
        if term.factors.len() < 2 { continue; }

        let n = pr.info.n_factors;
        let full_mask: FactorSubset = (1u64 << n) - 1;
        let interm = &pr.memoir[&(full_mask, 0)];

        for (eval_idx, eval) in interm.evals.iter().enumerate() {
            let mut left_ext = pr.info.ext_bits(eval.left);
            let mut right_ext = pr.info.ext_bits(eval.right);
            let mut left_subset = eval.left;
            let mut right_subset = eval.right;

            if left_ext > right_ext {
                std::mem::swap(&mut left_ext, &mut right_ext);
                std::mem::swap(&mut left_subset, &mut right_subset);
            }

            let mut sum_ranges: Vec<RangeId> = Vec::new();
            let mut m = eval.contracted_sums;
            while m != 0 {
                let bit = m.trailing_zeros() as usize;
                sum_ranges.push(term.sum_indices[bit].range);
                m &= m - 1;
            }
            sum_ranges.sort();

            let lsi = LastStepIndices { left_ext, right_ext, sums: sum_ranges };

            let left_sub = make_sub_term(term, left_subset);
            let right_sub = make_sub_term(term, right_subset);

            let left_factor_ids: HashSet<IndexId> = left_sub.factors.iter()
                .flat_map(|f| f.indices.iter().copied()).collect();
            let right_factor_ids: HashSet<IndexId> = right_sub.factors.iter()
                .flat_map(|f| f.indices.iter().copied()).collect();
            let contracted_ids: HashSet<IndexId> = term.sum_indices.iter()
                .map(|i| i.id)
                .filter(|id| left_factor_ids.contains(id) && right_factor_ids.contains(id))
                .collect();

            let edge_info = EdgeInfo {
                term_idx,
                eval_idx,
                coeff: term.coeff.clone(),
                exc_cost: eval.cost - interm.best_cost,
            };

            // Left-first sweep: contracted indices canonicalized by left sub-term.
            {
                let (left_canon, contracted_map) = canonicalize_sub_term(
                    &left_sub, &ext_ids, &ext_range, &contracted_ids, None, true, &pool, comp.tensors(),
                );
                let (right_canon, _) = canonicalize_sub_term(
                    &right_sub, &ext_ids, &ext_range, &contracted_ids, Some(&contracted_map), false, &pool, comp.tensors(),
                );
                groups_left.entry(lsi.clone()).or_default().push((left_canon, right_canon, edge_info.clone()));
            }
            // Right-first sweep: contracted indices canonicalized by right sub-term.
            {
                let (right_canon, contracted_map) = canonicalize_sub_term(
                    &right_sub, &ext_ids, &ext_range, &contracted_ids, None, false, &pool, comp.tensors(),
                );
                let (left_canon, _) = canonicalize_sub_term(
                    &left_sub, &ext_ids, &ext_range, &contracted_ids, Some(&contracted_map), true, &pool, comp.tensors(),
                );
                groups_right.entry(lsi.clone()).or_default().push((left_canon, right_canon, edge_info));
            }
        }
    }

    let mut result: Vec<ConstrGraph> = groups_left.into_iter().chain(groups_right.into_iter())
        .map(|(lsi, entries)| {
            let mut vertex_map: HashMap<(Term, Side), VertexId> = HashMap::new();
            let mut vertices: Vec<Term> = Vec::new();
            let mut vertex_sides: Vec<Side> = Vec::new();
            let mut edges = Vec::new();

            for (left_canon, right_canon, edge_info) in entries {
                let left_vid = ensure_vertex(&mut vertex_map, &mut vertices, &mut vertex_sides, left_canon, Side::Left);
                let right_vid = ensure_vertex(&mut vertex_map, &mut vertices, &mut vertex_sides, right_canon, Side::Right);
                edges.push((left_vid, right_vid, edge_info));
            }

            ConstrGraph { vertices, vertex_side: vertex_sides, edges, last_step: lsi }
        })
        .collect();

    result.sort_by(|a, b| {
        a.last_step.left_ext.cmp(&b.last_step.left_ext)
            .then(a.last_step.right_ext.cmp(&b.last_step.right_ext))
            .then(a.last_step.sums.cmp(&b.last_step.sums))
    });

    result
}

fn ensure_vertex(
    map: &mut HashMap<(Term, Side), VertexId>,
    verts: &mut Vec<Term>,
    sides: &mut Vec<Side>,
    canon: Term,
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

    // Build IndexId -> RangeId map from all sum_indices in def
    let mut id_to_range: HashMap<IndexId, RangeId> = HashMap::new();
    for term in &def.terms {
        for idx in &term.sum_indices {
            id_to_range.insert(idx.id, idx.range);
        }
    }

    for graph in &graphs {
        if graph.edges.is_empty() { continue; }

        let first_term_idx = graph.edges[0].2.term_idx;
        let info = &parenth_results[first_term_idx].info;
        let coeffs = compute_cost_coeffs(&graph.last_step, info, comp.ranges());
        let bicliques = find_bicliques(graph, &coeffs);

        for bc in bicliques {
            if bc.saving <= 0 { continue; }

            let is_complete = bc.left_verts.iter().all(|(lv, _)| {
                bc.right_verts.iter().all(|(rv, _)| !graph.edges_between(*lv, *rv).is_empty())
            });
            if !is_complete { continue; }

            let mut left_ids: Vec<VertexId> = bc.left_verts.iter().map(|(v, _)| *v).collect();
            let mut right_ids: Vec<VertexId> = bc.right_verts.iter().map(|(v, _)| *v).collect();
            left_ids.sort();
            right_ids.sort();
            if !seen_vertex_sets.insert((left_ids, right_ids)) { continue; }

            let terms_consumed = bits_to_vec(bc.terms_used);

            let (sample_lv, _) = &bc.left_verts[0];
            let (sample_rv, _) = &bc.right_verts[0];
            let contracted = contracted_indices(graph, *sample_lv, *sample_rv, &id_to_range);
            let contracted_ids: HashSet<IndexId> = contracted.iter().map(|i| i.id).collect();

            let left_ext = bits_to_indices(graph.last_step.left_ext, &def.ext_indices);
            let right_ext = bits_to_indices(graph.last_step.right_ext, &def.ext_indices);

            // Compute correct intermediate coefficients from edge data.
            // The intermediate term coefficient is bc_side(v) * vterm(v).coeff.
            let leading_coeff = bc.leading_coeff.clone()
                .unwrap_or_else(|| Ratio::from_integer(1));

            // Left vertex coefficients: bc_left(k) * vterm.coeff
            let left_coeffs: Vec<Rational> = bc.left_verts.iter().map(|(lv, bc_coeff)| {
                bc_coeff * &graph.vertices[lv.0].coeff
            }).collect();

            // Right vertex coefficients: bc_right(m) * vterm.coeff
            let right_coeffs: Vec<Rational> = bc.right_verts.iter().map(|(rv, bc_coeff)| {
                bc_coeff * &graph.vertices[rv.0].coeff
            }).collect();

            let mut intermediates = Vec::new();
            let mut candidate_id = next_tensor_id.0;

            let (left_tid, left_indices) = build_side(
                &bc.left_verts, &left_coeffs, graph, &left_ext, &contracted, &contracted_ids,
                &mut intermediates, &mut candidate_id,
            );
            let (right_tid, right_indices) = build_side(
                &bc.right_verts, &right_coeffs, graph, &right_ext, &contracted, &contracted_ids,
                &mut intermediates, &mut candidate_id,
            );

            let coeff = leading_coeff;
            let replacement_term = Term {
                coeff,
                sum_indices: contracted,
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

/// Get contracted Index objects: indices appearing in both left and right vertex factors.
fn contracted_indices(
    graph: &ConstrGraph,
    left_v: VertexId,
    right_v: VertexId,
    id_to_range: &HashMap<IndexId, RangeId>,
) -> Vec<Index> {
    let left_term = &graph.vertices[left_v.0];
    let right_term = &graph.vertices[right_v.0];
    let right_factor_ids: HashSet<IndexId> = right_term.factors.iter()
        .flat_map(|f| f.indices.iter().copied()).collect();
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for f in &left_term.factors {
        for &id in &f.indices {
            if right_factor_ids.contains(&id) && seen.insert(id) {
                if let Some(&range) = id_to_range.get(&id) {
                    result.push(Index { id, range });
                }
            }
        }
    }
    result
}

/// Build an intermediate TensorDef for one side of a biclique.
/// Returns (TensorId, Vec<IndexId>) for use as a factor in the replacement term.
/// `coeffs` is a pre-computed coefficient for each vertex (same order as `verts`).
fn build_side(
    verts: &[(VertexId, Rational)],
    coeffs: &[Rational],
    graph: &ConstrGraph,
    side_ext: &[Index],
    contracted: &[Index],
    contracted_ids: &HashSet<IndexId>,
    intermediates: &mut Vec<TensorDef>,
    next_id: &mut u32,
) -> (TensorId, Vec<IndexId>) {
    let interm_ext: Vec<Index> = side_ext.iter().chain(contracted.iter()).cloned().collect();
    let interm_idx_ids: Vec<IndexId> = interm_ext.iter().map(|i| i.id).collect();

    if verts.len() == 1 {
        let (v, _) = &verts[0];
        let coeff = &coeffs[0];
        let vterm = &graph.vertices[v.0];
        if vterm.factors.len() == 1 && vterm.sum_indices.is_empty() && *coeff == Ratio::from_integer(1) {
            let f = &vterm.factors[0];
            return (f.tensor, f.indices.clone());
        }
        let tid = TensorId(*next_id);
        *next_id += 1;
        intermediates.push(TensorDef {
            base: tid,
            ext_indices: interm_ext,
            terms: vec![Term {
                coeff: coeff.clone(),
                sum_indices: vterm.sum_indices.iter()
                    .filter(|idx| !contracted_ids.contains(&idx.id))
                    .cloned().collect(),
                factors: vterm.factors.clone(),
            }],
        });
        return (tid, interm_idx_ids);
    }

    let tid = TensorId(*next_id);
    *next_id += 1;
    let terms: Vec<Term> = verts.iter().zip(coeffs.iter()).map(|((v, _), coeff)| {
        let vterm = &graph.vertices[v.0];
        Term {
            coeff: coeff.clone(),
            sum_indices: vterm.sum_indices.iter()
                .filter(|idx| !contracted_ids.contains(&idx.id))
                .cloned().collect(),
            factors: vterm.factors.clone(),
        }
    }).collect();
    intermediates.push(TensorDef { base: tid, ext_indices: interm_ext, terms });
    (tid, interm_idx_ids)
}
