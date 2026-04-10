use crate::repr::{Range, TensorComputation, TensorDef};

/// FLOP cost of evaluating one TensorDef.
///
/// For each term: contraction cost + addition into output.
/// - No summation: ext_size (copy/scale) + ext_size (addition)
/// - With summation: 2 * ext_size * sum_size (multiply-add) + ext_size (addition)
pub fn def_cost(def: &TensorDef, ranges: &[Range]) -> u64 {
    let ext_size: u64 = def
        .ext_indices
        .iter()
        .map(|idx| ranges[idx.range.0 as usize].size)
        .product::<u64>()
        .max(1);

    def.terms
        .iter()
        .map(|term| {
            let sum_size: u64 = term
                .sum_indices
                .iter()
                .map(|idx| ranges[idx.range.0 as usize].size)
                .product::<u64>()
                .max(1);

            let contraction = if sum_size == 1 {
                ext_size
            } else {
                2 * ext_size * sum_size
            };
            contraction + ext_size
        })
        .sum()
}

/// Total FLOP cost of an entire computation.
pub fn total_cost(comp: &TensorComputation) -> u64 {
    comp.definitions()
        .iter()
        .map(|def| def_cost(def, comp.ranges()))
        .sum()
}
