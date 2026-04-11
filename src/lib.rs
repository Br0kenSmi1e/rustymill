pub mod repr;
pub mod cost;
pub mod canon;
pub mod parenth;
pub mod constr;

// Re-export primary types at crate root for convenience.
pub use repr::TensorComputation;
pub use cost::{def_cost, total_cost};
pub use canon::{canon_term, CanonTerm, CanonFactor, CanonIndex};
pub use parenth::{parenthesize, extract_optimal, ParenthResult, IndexInfo, Eval, Interm};
