use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::repr::{
    Factor, Index, IndexId, RangeId, Rational, SymAction, SymGenerator, TensorDef, TensorId,
    TensorInfo, Term,
};
use crate::rl_parenth::TermSplit;

pub struct CanonDefContext {
    ext_ids: HashSet<IndexId>,
    ext_range: HashMap<IndexId, RangeId>,
    pool: HashMap<RangeId, Vec<IndexId>>,
    tensors: Vec<TensorInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonSplitPair {
    pub left_assigned: TermSplit,
    pub right_assigned: TermSplit,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CanonicalKey {
    coeff_num: i64,
    coeff_den: i64,
    factors: Vec<(u32, Vec<u32>)>,
}

#[derive(Clone, Copy)]
enum SplitSide {
    Left,
    Right,
}

struct CanonEnv {
    dummy_range: HashMap<IndexId, RangeId>,
}

#[derive(Clone)]
struct StructuralFactor {
    tensor: TensorId,
    indices: Vec<IndexId>,
}

#[derive(Clone)]
struct TermSkeleton {
    coeff: Rational,
    factors: Vec<SkeletonFactor>,
}

#[derive(Clone)]
struct SkeletonFactor {
    tensor: TensorId,
    indices: Vec<SkeletonIndex>,
}

#[derive(Clone, Copy)]
enum SkeletonIndex {
    External(IndexId),
    LocalDummy { original: IndexId, range: RangeId },
}

pub fn build_canon_def_context(def: &TensorDef, tensors: &[TensorInfo]) -> CanonDefContext {
    CanonDefContext {
        ext_ids: def.ext_indices.iter().map(|index| index.id).collect(),
        ext_range: def
            .ext_indices
            .iter()
            .map(|index| (index.id, index.range))
            .collect(),
        pool: build_canon_pool(def),
        tensors: tensors.to_vec(),
    }
}

pub fn canon_term(term: &Term, cx: &CanonDefContext) -> Term {
    let env = CanonEnv {
        dummy_range: term
            .sum_indices
            .iter()
            .map(|index| (index.id, index.range))
            .collect(),
    };
    let skeletons = enumerate_term_skeletons(term, cx, &env);

    skeletons
        .iter()
        .map(|skeleton| rename_standalone_skeleton(term, skeleton, cx))
        .min_by(term_cmp)
        .expect("canonicalization should produce at least one candidate")
}

pub fn canon_split(split: &TermSplit, cx: &CanonDefContext) -> CanonSplitPair {
    let shared_ids = shared_ids_for_split(split);
    let (left_owner_term, left_shared_map) =
        canon_owner_term(&split.left_sub_term, SplitSide::Left, &shared_ids, cx);
    let right_follower_term = canon_follower_term(
        &split.right_sub_term,
        SplitSide::Right,
        &shared_ids,
        &left_shared_map,
        cx,
    );
    let (right_owner_term, right_shared_map) =
        canon_owner_term(&split.right_sub_term, SplitSide::Right, &shared_ids, cx);
    let left_follower_term = canon_follower_term(
        &split.left_sub_term,
        SplitSide::Left,
        &shared_ids,
        &right_shared_map,
        cx,
    );

    CanonSplitPair {
        left_assigned: TermSplit {
            left_sub_term: left_owner_term,
            right_sub_term: right_follower_term,
            last_step: split.last_step.clone(),
        },
        right_assigned: TermSplit {
            left_sub_term: left_follower_term,
            right_sub_term: right_owner_term,
            last_step: split.last_step.clone(),
        },
    }
}

pub fn canonical_term_key(term: &Term) -> CanonicalKey {
    CanonicalKey {
        coeff_num: *term.coeff.numer(),
        coeff_den: *term.coeff.denom(),
        factors: term
            .factors
            .iter()
            .map(|factor| {
                (
                    factor.tensor.0,
                    factor.indices.iter().map(|index| index.0).collect(),
                )
            })
            .collect(),
    }
}

fn build_canon_pool(def: &TensorDef) -> HashMap<RangeId, Vec<IndexId>> {
    let mut pool: HashMap<RangeId, Vec<IndexId>> = HashMap::new();
    for term in &def.terms {
        for index in &term.sum_indices {
            pool.entry(index.range).or_default().push(index.id);
        }
    }
    for indices in pool.values_mut() {
        indices.sort();
        indices.dedup();
    }
    pool
}

fn enumerate_term_skeletons(
    term: &Term,
    cx: &CanonDefContext,
    env: &CanonEnv,
) -> Vec<TermSkeleton> {
    let sym_groups: Vec<Vec<(Vec<usize>, SymAction)>> = term
        .factors
        .iter()
        .map(|factor| {
            enumerate_sym_group(
                &tensor_symmetry(cx, factor.tensor).symmetry,
                factor.indices.len(),
            )
        })
        .collect();

    let combo_sizes: Vec<usize> = sym_groups.iter().map(Vec::len).collect();
    let mut combo = vec![0usize; term.factors.len()];
    let mut skeletons = Vec::new();

    loop {
        let (coeff, mut factors) = build_symmetry_applied_factors(term, &sym_groups, &combo);
        sort_structural_factors(&mut factors, cx, env);

        let groups = tied_groups(&factors, cx, env);
        let mut group_perms: Vec<Vec<usize>> =
            groups.iter().map(|group| (0..group.len()).collect()).collect();

        loop {
            skeletons.push(TermSkeleton {
                coeff,
                factors: materialize_factor_order(&factors, &groups, &group_perms)
                    .into_iter()
                    .map(|factor| skeletonize_factor(factor, cx, env))
                    .collect(),
            });

            if !advance_group_perms(&mut group_perms) {
                break;
            }
        }

        if !advance_choice_vector(&mut combo, &combo_sizes) {
            break;
        }
    }

    skeletons
}

fn rename_standalone_skeleton(
    original: &Term,
    skeleton: &TermSkeleton,
    cx: &CanonDefContext,
) -> Term {
    let mut remap = HashMap::new();
    let mut counters: HashMap<RangeId, usize> = HashMap::new();

    for factor in &skeleton.factors {
        for &index in &factor.indices {
            match index {
                SkeletonIndex::External(id) => {
                    remap.entry(id).or_insert(id);
                }
                SkeletonIndex::LocalDummy { original, range } => {
                    remap.entry(original).or_insert_with(|| {
                        let slot = counters.entry(range).or_insert(0);
                        let canonical = cx.pool[&range][*slot];
                        *slot += 1;
                        canonical
                    });
                }
            }
        }
    }

    let factors = skeleton
        .factors
        .iter()
        .map(|factor| Factor {
            tensor: factor.tensor,
            indices: factor
                .indices
                .iter()
                .map(|index| match *index {
                    SkeletonIndex::External(id) => id,
                    SkeletonIndex::LocalDummy { original, .. } => remap[&original],
                })
                .collect(),
        })
        .collect();

    let mut sum_indices: Vec<Index> = original
        .sum_indices
        .iter()
        .map(|index| Index {
            id: remap[&index.id],
            range: index.range,
        })
        .collect();
    sum_indices.sort_by_key(|index| (index.range.0, index.id.0));

    Term {
        coeff: skeleton.coeff,
        sum_indices,
        factors,
    }
}

fn shared_ids_for_split(split: &TermSplit) -> HashSet<IndexId> {
    let left_ids: HashSet<IndexId> = split
        .left_sub_term
        .factors
        .iter()
        .flat_map(|factor| factor.indices.iter().copied())
        .collect();
    let right_ids: HashSet<IndexId> = split
        .right_sub_term
        .factors
        .iter()
        .flat_map(|factor| factor.indices.iter().copied())
        .collect();

    left_ids.intersection(&right_ids).copied().collect()
}

fn canon_owner_term(
    term: &Term,
    side: SplitSide,
    shared_ids: &HashSet<IndexId>,
    cx: &CanonDefContext,
) -> (Term, HashMap<IndexId, IndexId>) {
    let env = CanonEnv {
        dummy_range: term
            .sum_indices
            .iter()
            .map(|index| (index.id, index.range))
            .collect(),
    };

    enumerate_term_skeletons(term, cx, &env)
        .into_iter()
        .map(|skeleton| rename_owner_skeleton(term, &skeleton, side, shared_ids, cx))
        .min_by(owner_candidate_cmp)
        .expect("split owner canonicalization should produce at least one candidate")
}

fn canon_follower_term(
    term: &Term,
    side: SplitSide,
    shared_ids: &HashSet<IndexId>,
    shared_map: &HashMap<IndexId, IndexId>,
    cx: &CanonDefContext,
) -> Term {
    let env = CanonEnv {
        dummy_range: term
            .sum_indices
            .iter()
            .map(|index| (index.id, index.range))
            .collect(),
    };

    enumerate_term_skeletons(term, cx, &env)
        .into_iter()
        .map(|skeleton| rename_follower_skeleton(term, &skeleton, side, shared_ids, shared_map, cx))
        .min_by(term_cmp)
        .expect("split follower canonicalization should produce at least one candidate")
}

fn rename_owner_skeleton(
    original: &Term,
    skeleton: &TermSkeleton,
    side: SplitSide,
    shared_ids: &HashSet<IndexId>,
    cx: &CanonDefContext,
) -> (Term, HashMap<IndexId, IndexId>) {
    let shared_counts = split_shared_counts(original, shared_ids);
    let mut shared_map = HashMap::new();
    let mut shared_counters: HashMap<RangeId, usize> = HashMap::new();
    let mut left_private_counters = shared_counts.clone();
    let mut right_private_counters = split_right_private_starts(cx, &shared_counts);

    for factor in &skeleton.factors {
        for &index in &factor.indices {
            match index {
                SkeletonIndex::External(_) => {}
                SkeletonIndex::LocalDummy { original, range } => {
                    if shared_ids.contains(&original) {
                        shared_map.entry(original).or_insert_with(|| {
                            let slot = shared_counters.entry(range).or_insert(0);
                            let canonical = cx.pool[&range][*slot];
                            *slot += 1;
                            canonical
                        });
                    } else {
                        allocate_private_dummy(
                            original,
                            range,
                            side,
                            &mut shared_map,
                            &mut left_private_counters,
                            &mut right_private_counters,
                            cx,
                        );
                    }
                }
            }
        }
    }

    (
        rebuild_split_term(original, skeleton, shared_ids, &shared_map),
        shared_map
            .into_iter()
            .filter(|(original, _)| shared_ids.contains(original))
            .collect(),
    )
}

fn rename_follower_skeleton(
    original: &Term,
    skeleton: &TermSkeleton,
    side: SplitSide,
    shared_ids: &HashSet<IndexId>,
    shared_map: &HashMap<IndexId, IndexId>,
    cx: &CanonDefContext,
) -> Term {
    let shared_counts = split_shared_counts(original, shared_ids);
    let mut remap = shared_map.clone();
    let mut left_private_counters = shared_counts.clone();
    let mut right_private_counters = split_right_private_starts(cx, &shared_counts);

    for factor in &skeleton.factors {
        for &index in &factor.indices {
            match index {
                SkeletonIndex::External(id) => {
                    remap.entry(id).or_insert(id);
                }
                SkeletonIndex::LocalDummy { original, range } => {
                    if !shared_ids.contains(&original) {
                        allocate_private_dummy(
                            original,
                            range,
                            side,
                            &mut remap,
                            &mut left_private_counters,
                            &mut right_private_counters,
                            cx,
                        );
                    }
                }
            }
        }
    }

    rebuild_split_term(original, skeleton, shared_ids, &remap)
}

fn split_shared_counts(original: &Term, shared_ids: &HashSet<IndexId>) -> HashMap<RangeId, usize> {
    let mut counts = HashMap::new();
    for index in &original.sum_indices {
        if shared_ids.contains(&index.id) {
            *counts.entry(index.range).or_insert(0) += 1;
        }
    }
    counts
}

fn split_right_private_starts(
    cx: &CanonDefContext,
    shared_counts: &HashMap<RangeId, usize>,
) -> HashMap<RangeId, usize> {
    shared_counts
        .keys()
        .map(|&range| (range, cx.pool[&range].len()))
        .collect()
}

fn allocate_private_dummy(
    original: IndexId,
    range: RangeId,
    side: SplitSide,
    remap: &mut HashMap<IndexId, IndexId>,
    left_private_counters: &mut HashMap<RangeId, usize>,
    right_private_counters: &mut HashMap<RangeId, usize>,
    cx: &CanonDefContext,
) {
    remap.entry(original).or_insert_with(|| match side {
        SplitSide::Left => {
            let slot = left_private_counters.entry(range).or_insert(0);
            let canonical = cx.pool[&range][*slot];
            *slot += 1;
            canonical
        }
        SplitSide::Right => {
            let slot = right_private_counters
                .entry(range)
                .or_insert_with(|| cx.pool[&range].len());
            *slot -= 1;
            cx.pool[&range][*slot]
        }
    });
}

fn rebuild_split_term(
    original: &Term,
    skeleton: &TermSkeleton,
    shared_ids: &HashSet<IndexId>,
    remap: &HashMap<IndexId, IndexId>,
) -> Term {
    let factors = skeleton
        .factors
        .iter()
        .map(|factor| Factor {
            tensor: factor.tensor,
            indices: factor
                .indices
                .iter()
                .map(|index| match *index {
                    SkeletonIndex::External(id) => id,
                    SkeletonIndex::LocalDummy { original, .. } => remap[&original],
                })
                .collect(),
        })
        .collect();

    let mut sum_indices: Vec<Index> = original
        .sum_indices
        .iter()
        .filter(|index| !shared_ids.contains(&index.id))
        .map(|index| Index {
            id: remap.get(&index.id).copied().unwrap_or(index.id),
            range: index.range,
        })
        .collect();
    sum_indices.sort_by_key(|index| (index.range.0, index.id.0));

    Term {
        coeff: skeleton.coeff,
        sum_indices,
        factors,
    }
}

fn owner_candidate_cmp(
    left: &(Term, HashMap<IndexId, IndexId>),
    right: &(Term, HashMap<IndexId, IndexId>),
) -> Ordering {
    term_cmp(&left.0, &right.0).then(shared_map_key(&left.1).cmp(&shared_map_key(&right.1)))
}

fn shared_map_key(shared_map: &HashMap<IndexId, IndexId>) -> Vec<(u32, u32)> {
    let mut key: Vec<(u32, u32)> = shared_map
        .iter()
        .map(|(original, canonical)| (original.0, canonical.0))
        .collect();
    key.sort();
    key
}

fn build_symmetry_applied_factors(
    term: &Term,
    sym_groups: &[Vec<(Vec<usize>, SymAction)>],
    combo: &[usize],
) -> (Rational, Vec<StructuralFactor>) {
    let coeff_action = term
        .factors
        .iter()
        .enumerate()
        .fold(SymAction::Identity, |acc, (i, _)| {
            acc.combine(sym_groups[i][combo[i]].1)
        });
    let coeff = apply_action_to_coeff(term.coeff, coeff_action);

    let factors = term
        .factors
        .iter()
        .enumerate()
        .map(|(i, factor)| {
            let (perm, _) = &sym_groups[i][combo[i]];
            StructuralFactor {
                tensor: factor.tensor,
                indices: perm.iter().map(|&position| factor.indices[position]).collect(),
            }
        })
        .collect();

    (coeff, factors)
}

fn sort_structural_factors(
    factors: &mut [StructuralFactor],
    cx: &CanonDefContext,
    env: &CanonEnv,
) {
    factors.sort_by_key(|factor| structural_factor_key(factor, cx, env));
}

fn tied_groups(
    factors: &[StructuralFactor],
    cx: &CanonDefContext,
    env: &CanonEnv,
) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut start = 0;
    while start < factors.len() {
        let key = structural_factor_key(&factors[start], cx, env);
        let mut end = start + 1;
        while end < factors.len() && structural_factor_key(&factors[end], cx, env) == key {
            end += 1;
        }
        groups.push((start..end).collect());
        start = end;
    }
    groups
}

fn materialize_factor_order(
    factors: &[StructuralFactor],
    groups: &[Vec<usize>],
    group_perms: &[Vec<usize>],
) -> Vec<StructuralFactor> {
    groups
        .iter()
        .zip(group_perms.iter())
        .flat_map(|(group, perm)| perm.iter().map(|&position| factors[group[position]].clone()))
        .collect()
}

fn skeletonize_factor(
    factor: StructuralFactor,
    cx: &CanonDefContext,
    env: &CanonEnv,
) -> SkeletonFactor {
    SkeletonFactor {
        tensor: factor.tensor,
        indices: factor
            .indices
            .into_iter()
            .map(|id| {
                if cx.ext_ids.contains(&id) {
                    SkeletonIndex::External(id)
                } else {
                    SkeletonIndex::LocalDummy {
                        original: id,
                        range: env.dummy_range[&id],
                    }
                }
            })
            .collect(),
    }
}

fn structural_factor_key(
    factor: &StructuralFactor,
    cx: &CanonDefContext,
    env: &CanonEnv,
) -> (TensorId, Vec<(u8, RangeId)>) {
    let indices = factor
        .indices
        .iter()
        .map(|&id| {
            if cx.ext_ids.contains(&id) {
                (0, cx.ext_range[&id])
            } else {
                (1, env.dummy_range[&id])
            }
        })
        .collect();
    (factor.tensor, indices)
}

fn tensor_symmetry(cx: &CanonDefContext, tensor: TensorId) -> &TensorInfo {
    cx.tensors
        .iter()
        .find(|info| info.id == tensor)
        .unwrap_or_else(|| panic!("missing tensor info for tensor {}", tensor.0))
}

fn enumerate_sym_group(generators: &[SymGenerator], arity: usize) -> Vec<(Vec<usize>, SymAction)> {
    let identity: Vec<usize> = (0..arity).collect();
    let mut elements = vec![(identity.clone(), SymAction::Identity)];
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    seen.insert(identity);

    let mut cursor = 0;
    while cursor < elements.len() {
        let (perm, action) = elements[cursor].clone();
        cursor += 1;

        for generator in generators {
            let new_perm: Vec<usize> = generator.perm.iter().map(|&index| perm[index]).collect();
            if seen.insert(new_perm.clone()) {
                elements.push((new_perm, action.combine(generator.action)));
            }
        }
    }

    elements
}

fn apply_action_to_coeff(coeff: Rational, action: SymAction) -> Rational {
    match action {
        SymAction::Identity => coeff,
        SymAction::Negate => -coeff,
        SymAction::Conjugate => coeff,
        SymAction::NegateConjugate => -coeff,
    }
}

fn term_cmp(a: &Term, b: &Term) -> Ordering {
    let lhs = (*a.coeff.numer() as i128) * (*b.coeff.denom() as i128);
    let rhs = (*b.coeff.numer() as i128) * (*a.coeff.denom() as i128);
    if lhs != rhs {
        return lhs.cmp(&rhs);
    }

    for (left_factor, right_factor) in a.factors.iter().zip(b.factors.iter()) {
        if left_factor.tensor != right_factor.tensor {
            return left_factor.tensor.0.cmp(&right_factor.tensor.0);
        }
        for (left_index, right_index) in left_factor.indices.iter().zip(right_factor.indices.iter())
        {
            if left_index != right_index {
                return left_index.0.cmp(&right_index.0);
            }
        }
        if left_factor.indices.len() != right_factor.indices.len() {
            return left_factor.indices.len().cmp(&right_factor.indices.len());
        }
    }
    if a.factors.len() != b.factors.len() {
        return a.factors.len().cmp(&b.factors.len());
    }

    for (left_index, right_index) in a.sum_indices.iter().zip(b.sum_indices.iter()) {
        if left_index.range != right_index.range {
            return left_index.range.0.cmp(&right_index.range.0);
        }
        if left_index.id != right_index.id {
            return left_index.id.0.cmp(&right_index.id.0);
        }
    }

    a.sum_indices.len().cmp(&b.sum_indices.len())
}

fn next_permutation(perm: &mut [usize]) -> bool {
    let len = perm.len();
    if len <= 1 {
        return false;
    }

    let mut i = len - 1;
    while i > 0 && perm[i - 1] >= perm[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }

    let pivot = i - 1;
    let mut j = len - 1;
    while perm[j] <= perm[pivot] {
        j -= 1;
    }

    perm.swap(pivot, j);
    perm[i..].reverse();
    true
}

fn advance_group_perms(group_perms: &mut [Vec<usize>]) -> bool {
    for perm in group_perms.iter_mut().rev() {
        if next_permutation(perm) {
            return true;
        }
        *perm = (0..perm.len()).collect();
    }
    false
}

fn advance_choice_vector(choice: &mut [usize], sizes: &[usize]) -> bool {
    if choice.is_empty() {
        return false;
    }

    for position in (0..choice.len()).rev() {
        choice[position] += 1;
        if choice[position] < sizes[position] {
            return true;
        }
        choice[position] = 0;
    }

    false
}
