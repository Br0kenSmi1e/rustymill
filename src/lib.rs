pub mod biclique;
pub mod biclique_action;
pub mod repr;
pub mod cost;
pub mod canon;
pub mod rl_canon;
pub mod parenth;
pub mod rl_parenth;
pub mod constr;
pub mod mcts;
pub mod optimize;
pub mod convert;

// Re-export primary types at crate root for convenience.
pub use repr::TensorComputation;
pub use cost::{def_cost, total_cost};
pub use canon::{canon_term, CanonTerm, CanonFactor, CanonIndex};
pub use parenth::{parenthesize, extract_optimal, ParenthResult, IndexInfo, Eval, Interm};
pub use constr::{factorizations, Factorization};
pub use optimize::{greedy_optimize, apply_factorization, mcts_optimize};
pub use convert::{read_json, write_json, from_json, to_json};
