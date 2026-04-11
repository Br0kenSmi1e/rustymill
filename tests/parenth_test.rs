use rustymill::repr::*;
use rustymill::parenth::*;
use num::rational::Ratio;

/// Helper: build a term A[a,c] * B[c,d] * C[d,b]
/// ext: a(occ=10), b(virt=100)
/// sum: c(occ=10), d(virt=100)
fn make_abc_term() -> (TensorComputation, Term, Vec<Index>) {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let virt = comp.add_range(100);
    let _a_tensor = comp.add_tensor(&[occ, occ], vec![]);
    let _b_tensor = comp.add_tensor(&[occ, virt], vec![]);
    let _c_tensor = comp.add_tensor(&[virt, virt], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);
    let d = IndexId(3);

    let ext_indices = vec![
        Index { id: a, range: occ },
        Index { id: b, range: virt },
    ];

    let term = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![
            Index { id: c, range: occ },
            Index { id: d, range: virt },
        ],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a, c] },
            Factor { tensor: TensorId(1), indices: vec![c, d] },
            Factor { tensor: TensorId(2), indices: vec![d, b] },
        ],
    };

    (comp, term, ext_indices)
}

#[test]
fn test_precompute_index_info() {
    let (comp, term, ext_indices) = make_abc_term();
    let info = IndexInfo::new(&term, &ext_indices, comp.ranges());

    assert_eq!(info.n_factors, 3);
    assert_eq!(info.sum_sizes.len(), 2);
    assert_eq!(info.ext_sizes.len(), 2);

    assert_eq!(info.factor_sum_indices[0], 0b01);
    assert_eq!(info.factor_ext_indices[0], 0b01);

    assert_eq!(info.factor_sum_indices[1], 0b11);
    assert_eq!(info.factor_ext_indices[1], 0b00);

    assert_eq!(info.factor_sum_indices[2], 0b10);
    assert_eq!(info.factor_ext_indices[2], 0b10);
}

#[test]
fn test_subset_index_bits() {
    let (comp, term, ext_indices) = make_abc_term();
    let info = IndexInfo::new(&term, &ext_indices, comp.ranges());

    let s: FactorSubset = 0b011;
    assert_eq!(info.sum_bits(s), 0b11);
    assert_eq!(info.ext_bits(s), 0b01);

    let s: FactorSubset = 0b110;
    assert_eq!(info.sum_bits(s), 0b11);
    assert_eq!(info.ext_bits(s), 0b10);

    let s: FactorSubset = 0b001;
    assert_eq!(info.sum_bits(s), 0b01);
    assert_eq!(info.ext_bits(s), 0b01);
}

#[test]
fn test_bitmask_size_product() {
    let (comp, term, ext_indices) = make_abc_term();
    let info = IndexInfo::new(&term, &ext_indices, comp.ranges());

    assert_eq!(info.size_product_sum(0b01), 10);
    assert_eq!(info.size_product_sum(0b10), 100);
    assert_eq!(info.size_product_sum(0b11), 1000);

    assert_eq!(info.size_product_ext(0b01), 10);
    assert_eq!(info.size_product_ext(0b10), 100);
    assert_eq!(info.size_product_ext(0b11), 1000);
}

#[test]
fn test_split_cost_two_factors() {
    // t[a,b] = sum_c A[a,c] * B[c,b], occ=10
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _a = comp.add_tensor(&[occ, occ], vec![]);
    let _b = comp.add_tensor(&[occ, occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);

    let ext = vec![
        Index { id: a, range: occ },
        Index { id: b, range: occ },
    ];
    let term = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![Index { id: c, range: occ }],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a, c] },
            Factor { tensor: TensorId(1), indices: vec![c, b] },
        ],
    };

    let info = IndexInfo::new(&term, &ext, comp.ranges());
    let cost = split_cost(&info, 0b01, 0b10);

    // contracted = c, ext = {a,b}, ext_size=100, sum_size=10
    // step = 2*100*10 + 100 = 2100
    assert_eq!(cost.step_cost, 2100);
    assert_eq!(cost.contracted_sums, 0b01);
}

#[test]
fn test_split_cost_no_contraction() {
    // A[a] * B[b], no sum indices
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _a = comp.add_tensor(&[occ], vec![]);
    let _b = comp.add_tensor(&[occ], vec![]);

    let a = IndexId(0);
    let b = IndexId(1);

    let ext = vec![
        Index { id: a, range: occ },
        Index { id: b, range: occ },
    ];
    let term = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a] },
            Factor { tensor: TensorId(1), indices: vec![b] },
        ],
    };

    let info = IndexInfo::new(&term, &ext, comp.ranges());
    let cost = split_cost(&info, 0b01, 0b10);

    // No contracted sums, ext={a,b}, ext_size=100, sum_size=1
    // step = 100 + 100 = 200
    assert_eq!(cost.step_cost, 200);
    assert_eq!(cost.contracted_sums, 0);
}

#[test]
fn test_split_cost_partial_contraction() {
    // A[a,c]*B[c,d]*C[d,b] — split {A,B} vs {C}
    let (comp, term, ext_indices) = make_abc_term();
    let info = IndexInfo::new(&term, &ext_indices, comp.ranges());
    let cost = split_cost(&info, 0b011, 0b100);

    // contracted = d (bit 1), uncontracted = c (bit 0)
    // ext_of_result = {a,b}, uncontracted_sums = {c}
    // ext_size = 10 * 100 * 10 = 10000
    // sum_size = 100
    // step = 2 * 10000 * 100 + 10000 = 2_010_000
    assert_eq!(cost.step_cost, 2_010_000);
    assert_eq!(cost.contracted_sums, 0b10);
}
