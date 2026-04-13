use crate::constr::{factorizations, Factorization};
use crate::cost::total_cost;
use crate::mcts::{self, MctsState};
use crate::parenth::parenthesize;
use crate::repr::{TensorComputation, TensorDef};

/// Apply a factorization to a TensorComputation.
///
/// Replaces the definition at `def_index` with the factored version:
/// intermediates are inserted before it, consumed terms are removed,
/// and the replacement term is added.
///
/// Also registers the intermediate tensors in the computation.
pub fn apply_factorization(
    comp: &mut TensorComputation,
    def_index: usize,
    fact: &Factorization,
) {
    // Register intermediate tensors
    for interm in &fact.intermediates {
        let slots: Vec<_> = interm.ext_indices.iter().map(|idx| idx.range).collect();
        comp.add_tensor(&slots, vec![]);
    }

    // Modify the target definition: remove consumed terms, add replacement
    let def = &mut comp.definitions_mut()[def_index];
    let mut new_terms: Vec<_> = def
        .terms
        .iter()
        .enumerate()
        .filter(|(i, _)| !fact.terms_consumed.contains(i))
        .map(|(_, t)| t.clone())
        .collect();
    new_terms.push(fact.replacement_term.clone());
    def.terms = new_terms;

    // Insert intermediate definitions before the target definition
    let intermediates: Vec<TensorDef> = fact.intermediates.clone();
    for (i, interm) in intermediates.into_iter().enumerate() {
        comp.definitions_mut().insert(def_index + i, interm);
    }
}

/// Find the leftmost TensorDef (starting from `start_from`) that has
/// profitable factorizations. Returns all of them sorted by saving descending.
fn next_decision(
    comp: &TensorComputation,
    start_from: usize,
) -> Option<(usize, Vec<Factorization>)> {
    for (i, def) in comp.definitions().iter().enumerate().skip(start_from) {
        if def.terms.len() < 2 {
            continue;
        }

        let prs: Vec<_> = def
            .terms
            .iter()
            .map(|t| parenthesize(t, &def.ext_indices, comp.ranges()))
            .collect();

        let next_id = comp.next_tensor_id();
        let facts = factorizations(def, &prs, comp, next_id);

        let mut profitable: Vec<_> = facts.into_iter().filter(|f| f.saving > 0).collect();
        if !profitable.is_empty() {
            profitable.sort_by(|a, b| b.saving.cmp(&a.saving));
            return Some((i, profitable));
        }
    }
    None
}

/// Greedy optimization: repeatedly find and apply the best factorization
/// for the leftmost TensorDef with profitable bicliques, until none remain.
///
/// Returns the number of factorizations applied.
pub fn greedy_optimize(comp: &mut TensorComputation) -> usize {
    let mut count = 0;
    let mut start_from = 0;

    while let Some((def_idx, facts)) = next_decision(comp, start_from) {
        apply_factorization(comp, def_idx, &facts[0]);
        count += 1;
        start_from = def_idx;
    }

    count
}

// ---------------------------------------------------------------------------
// MCTS optimization
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FactorizationState {
    comp: TensorComputation,
    def_idx: usize,
    greedy_cost: f64,
    terminal: bool,
}

impl FactorizationState {
    fn new(comp: TensorComputation, start_from: usize, greedy_cost: f64) -> Self {
        match next_decision(&comp, start_from) {
            Some((def_idx, _)) => Self {
                comp,
                def_idx,
                greedy_cost,
                terminal: false,
            },
            None => Self {
                comp,
                def_idx: 0,
                greedy_cost,
                terminal: true,
            },
        }
    }
}

impl MctsState for FactorizationState {
    type Action = Factorization;

    fn available_actions(&self) -> Vec<Factorization> {
        if self.terminal {
            return Vec::new();
        }
        match next_decision(&self.comp, self.def_idx) {
            Some((_, facts)) => facts,
            None => Vec::new(),
        }
    }

    fn apply_action(&mut self, action: &Factorization) {
        apply_factorization(&mut self.comp, self.def_idx, action);
        match next_decision(&self.comp, self.def_idx) {
            Some((idx, _)) => self.def_idx = idx,
            None => self.terminal = true,
        }
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn rollout_reward(&self) -> f64 {
        let mut rollout = self.comp.clone();
        greedy_optimize(&mut rollout);
        let rollout_cost = total_cost(&rollout) as f64;
        (self.greedy_cost - rollout_cost) / self.greedy_cost
    }
}

/// MCTS optimization: use Monte Carlo Tree Search to explore factorization
/// sequences, trying to beat the greedy optimizer.
///
/// Returns the number of factorizations applied.
pub fn mcts_optimize(
    comp: &mut TensorComputation,
    iterations: u32,
    exploration: f64,
) -> usize {
    // Run greedy on a clone to get the baseline cost
    let mut greedy_clone = comp.clone();
    greedy_optimize(&mut greedy_clone);
    let greedy_cost = total_cost(&greedy_clone) as f64;

    let root = FactorizationState::new(comp.clone(), 0, greedy_cost);
    if root.terminal {
        return 0;
    }

    let actions = mcts::search(root, iterations, exploration);

    // Replay the best action sequence on comp
    let count = actions.len();
    for action in &actions {
        let (def_idx, _) = next_decision(comp, 0).expect("action replay failed");
        apply_factorization(comp, def_idx, action);
    }

    count
}
