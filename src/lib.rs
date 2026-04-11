pub mod repr;
pub mod cost;
pub mod canon;
pub mod parenth;

// Re-export primary types at crate root for convenience.
pub use repr::TensorComputation;
pub use cost::{def_cost, total_cost};
pub use canon::{canon_term, CanonTerm, CanonFactor, CanonIndex};
