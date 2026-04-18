//! Generic Monte Carlo Tree Search (UCT) engine.
//!
//! The MCTS algorithm is domain-agnostic. Domain-specific logic is provided
//! via the [`MctsState`] trait.

/// A state in the search space. Implementors define the action space,
/// transitions, terminal detection, and rollout policy.
pub trait MctsState: Clone {
    type Action: Clone;

    /// Return available actions, sorted best-first (most promising first).
    /// Empty means the state is terminal.
    fn available_actions(&self) -> Vec<Self::Action>;

    /// Apply an action to this state in place.
    /// The caller clones the state before calling this.
    fn apply_action(&mut self, action: &Self::Action);

    /// True if no more actions are possible.
    fn is_terminal(&self) -> bool;

    /// Run a rollout from this state and return a reward signal.
    fn rollout_reward(&self) -> f64;
}

struct MctsNode<S: MctsState> {
    state: S,
    actions: Vec<S::Action>,
    children: Vec<Option<usize>>, // parallel to actions; index into arena
    expanded: bool,
    visits: u32,
    total_reward: f64,
}

/// Run MCTS from the given root state and return the best action sequence.
///
/// - `iterations`: number of MCTS iterations (select → expand → rollout → backprop)
/// - `exploration`: UCB1 exploration constant (e.g. 1.414)
///
/// Returns the most-visited path from root to a terminal (or deepest explored) node.
pub fn search<S: MctsState>(root: S, iterations: u32, exploration: f64) -> Vec<S::Action> {
    let mut arena: Vec<MctsNode<S>> = Vec::new();

    // Create root node
    arena.push(MctsNode {
        state: root,
        actions: Vec::new(),
        children: Vec::new(),
        expanded: false,
        visits: 0,
        total_reward: 0.0,
    });

    for _ in 0..iterations {
        let mut path: Vec<(usize, usize)> = Vec::new(); // (node_idx, action_idx)
        let mut current = 0usize;

        // === SELECT ===
        loop {
            if !arena[current].expanded {
                break;
            }
            if arena[current].actions.is_empty() {
                // Terminal node
                break;
            }

            let action_idx = select_action(&arena, current, exploration);
            path.push((current, action_idx));

            match arena[current].children[action_idx] {
                Some(child_idx) => {
                    current = child_idx;
                }
                None => {
                    // Child not yet created — expand it
                    break;
                }
            }
        }

        // === EXPAND ===
        let leaf = if !arena[current].expanded {
            // Expand current node
            let actions = arena[current].state.available_actions();
            let n = actions.len();
            arena[current].actions = actions;
            arena[current].children = vec![None; n];
            arena[current].expanded = true;
            current
        } else if let Some(&(parent, action_idx)) = path.last() {
            // Create child node for the selected action
            let mut child_state = arena[parent].state.clone();
            child_state.apply_action(&arena[parent].actions[action_idx]);
            let actions = child_state.available_actions();
            let n = actions.len();
            let child_idx = arena.len();
            arena.push(MctsNode {
                state: child_state,
                actions,
                children: vec![None; n],
                expanded: true,
                visits: 0,
                total_reward: 0.0,
            });
            arena[parent].children[action_idx] = Some(child_idx);
            child_idx
        } else {
            // Root is terminal
            current
        };

        // === ROLLOUT ===
        let reward = arena[leaf].rollout_reward();

        // === BACKPROPAGATE ===
        arena[leaf].visits += 1;
        arena[leaf].total_reward += reward;
        for &(node_idx, _) in path.iter().rev() {
            arena[node_idx].visits += 1;
            arena[node_idx].total_reward += reward;
        }
    }

    // Extract best path: follow most-visited child at each level
    let best_path = extract_best_path(&arena);
    print_best_path_branching(&arena, 100);
    best_path
}

impl<S: MctsState> MctsNode<S> {
    fn rollout_reward(&self) -> f64 {
        self.state.rollout_reward()
    }
}

/// UCB1 selection: pick the action with highest UCB1 value.
/// Unvisited actions (in sorted order) are picked first.
fn select_action<S: MctsState>(
    arena: &[MctsNode<S>],
    node_idx: usize,
    exploration: f64,
) -> usize {
    let node = &arena[node_idx];
    let ln_parent = (node.visits as f64).ln();

    // First unvisited child (actions are sorted best-first)
    for (i, child) in node.children.iter().enumerate() {
        if child.is_none() {
            return i;
        }
    }

    // All visited — pick by UCB1
    let mut best_idx = 0;
    let mut best_ucb = f64::NEG_INFINITY;
    for (i, child_opt) in node.children.iter().enumerate() {
        if let Some(child_idx) = child_opt {
            let child = &arena[*child_idx];
            let mean = child.total_reward / child.visits as f64;
            let ucb = mean + exploration * (ln_parent / child.visits as f64).sqrt();
            if ucb > best_ucb {
                best_ucb = ucb;
                best_idx = i;
            }
        }
    }
    best_idx
}

/// Follow most-visited children from root to build the best action sequence.
fn extract_best_path<S: MctsState>(arena: &[MctsNode<S>]) -> Vec<S::Action> {
    let mut actions = Vec::new();
    let mut current = 0usize;

    loop {
        let node = &arena[current];
        if !node.expanded || node.actions.is_empty() {
            break;
        }

        // Pick the most-visited child
        let mut best_idx = 0;
        let mut best_visits = 0;
        for (i, child_opt) in node.children.iter().enumerate() {
            if let Some(child_idx) = child_opt {
                let visits = arena[*child_idx].visits;
                if visits > best_visits {
                    best_visits = visits;
                    best_idx = i;
                }
            }
        }

        if best_visits == 0 {
            break;
        }

        actions.push(node.actions[best_idx].clone());
        current = node.children[best_idx].unwrap();
    }

    actions
}

pub fn print_best_path_branching<S: MctsState>(arena: &[MctsNode<S>], max_depth: usize) {
    let mut current = 0usize;
    for depth in 0..=max_depth {
        if current >= arena.len() { break; }
        
        let node = &arena[current];
        let q_val = if node.visits > 0 { node.total_reward / node.visits as f64 } else { 0.0 };
        
        let branch_info = if node.expanded { 
            format!("{}", node.actions.len()) 
        } else { 
            "?".to_string() 
        };

        eprintln!("Depth {:>2}: visits={:<5} avgQ={:<7.3} | Actions: {}", depth, node.visits, q_val, branch_info);

        let mut best_child_idx: Option<usize> = None;
        let mut best_visits = 0u32;

        for child_opt in &node.children {
            if let Some(idx) = child_opt {
                let c_visits = arena[*idx].visits;
                if c_visits > best_visits {
                    best_visits = c_visits;
                    best_child_idx = Some(*idx);
                }
            }
        }

        match best_child_idx {
            Some(child_idx) => {
                eprintln!("  └─ (most visited) →");
                current = child_idx;
            }
            None => {
                eprintln!("  └─ [Leaf / Terminal / Unexplored]");
                break;
            }
        }
    }
}
