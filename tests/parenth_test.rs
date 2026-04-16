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
    let _a_tensor = comp.add_tensor(vec![]);
    let _b_tensor = comp.add_tensor(vec![]);
    let _c_tensor = comp.add_tensor(vec![]);

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
fn test_parenthesize_two_factors() {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _a = comp.add_tensor(vec![]);
    let _b = comp.add_tensor(vec![]);

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

    let result = parenthesize(&term, &ext, comp.ranges());

    let full = &result.memoir[&(0b11u64, 0)];
    assert_eq!(full.evals.len(), 1);
    assert_eq!(full.best_cost, 2000);
    assert_eq!(full.evals[0].left, 0b01);
    assert_eq!(full.evals[0].right, 0b10);

    assert_eq!(result.memoir[&(0b01u64, 0b01)].best_cost, 0);
    assert_eq!(result.memoir[&(0b10u64, 0b01)].best_cost, 0);
}

#[test]
fn test_parenthesize_three_factors() {
    let (comp, term, ext_indices) = make_abc_term();
    let result = parenthesize(&term, &ext_indices, comp.ranges());

    let full = &result.memoir[&(0b111u64, 0)];
    assert_eq!(full.evals.len(), 3);

    let min_cost = full.evals.iter().map(|e| e.cost).min().unwrap();
    assert_eq!(full.best_cost, min_cost);

    assert!(result.memoir.contains_key(&(0b011u64, 0b10)));
    assert!(result.memoir.contains_key(&(0b101u64, 0b11)));
    assert!(result.memoir.contains_key(&(0b110u64, 0b01)));
}

#[test]
fn test_parenthesize_optimal_order() {
    let (comp, term, ext_indices) = make_abc_term();
    let result = parenthesize(&term, &ext_indices, comp.ranges());

    let full = &result.memoir[&(0b111u64, 0)];
    assert_eq!(full.best_cost, 220_000);

    let worst = full.evals.iter().find(|e| {
        (e.left == 0b010 && e.right == 0b101) || (e.left == 0b101 && e.right == 0b010)
    }).unwrap();
    assert_eq!(worst.cost, 3_000_000);
}

#[test]
fn test_parenthesize_single_factor() {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let _a = comp.add_tensor(vec![]);
    let a = IndexId(0);
    let ext = vec![Index { id: a, range: occ }];
    let term = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![],
        factors: vec![Factor { tensor: TensorId(0), indices: vec![a] }],
    };

    let result = parenthesize(&term, &ext, comp.ranges());
    assert_eq!(result.memoir.len(), 1);
    assert_eq!(result.memoir[&(0b1u64, 0)].best_cost, 0);
    assert!(result.memoir[&(0b1u64, 0)].evals.is_empty());
}

#[test]
fn test_extract_optimal_two_factors() {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let a_tensor = comp.add_tensor(vec![]);
    let b_tensor = comp.add_tensor(vec![]);

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
            Factor { tensor: a_tensor, indices: vec![a, c] },
            Factor { tensor: b_tensor, indices: vec![c, b] },
        ],
    };

    let result = parenthesize(&term, &ext, comp.ranges());
    let defs = extract_optimal(&result, &term, &ext, &mut comp);

    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].terms.len(), 1);
    assert_eq!(defs[0].terms[0].factors.len(), 2);
    assert_eq!(defs[0].ext_indices.len(), 2);
    assert_eq!(defs[0].terms[0].sum_indices.len(), 1);
}

#[test]
fn test_extract_optimal_three_factors() {
    let (mut comp, term, ext_indices) = make_abc_term();
    let result = parenthesize(&term, &ext_indices, comp.ranges());
    let defs = extract_optimal(&result, &term, &ext_indices, &mut comp);

    assert_eq!(defs.len(), 2);

    for def in &defs {
        assert_eq!(def.terms.len(), 1);
        assert_eq!(def.terms[0].factors.len(), 2);
    }

    assert_eq!(defs.last().unwrap().ext_indices.len(), 2);
}

#[test]
fn test_parenthesize_stores_all_alternatives() {
    let (comp, term, ext_indices) = make_abc_term();
    let result = parenthesize(&term, &ext_indices, comp.ranges());

    let full = &result.memoir[&(0b111u64, 0)];
    assert_eq!(full.evals.len(), 3);

    let mut splits: Vec<(u64, u64)> = full.evals.iter()
        .map(|e| (e.left.min(e.right), e.left.max(e.right)))
        .collect();
    splits.sort();

    assert_eq!(splits, vec![
        (0b001, 0b110),
        (0b010, 0b101),
        (0b011, 0b100),
    ]);

    assert_eq!(result.memoir[&(0b110u64, 0b01)].evals.len(), 1);
}

#[test]
fn test_parenthesize_four_factors() {
    let mut comp = TensorComputation::new();
    let occ = comp.add_range(10);
    let virt = comp.add_range(100);
    let _a = comp.add_tensor(vec![]);
    let _b = comp.add_tensor(vec![]);
    let _c = comp.add_tensor(vec![]);
    let _d = comp.add_tensor(vec![]);

    let a = IndexId(0);
    let b = IndexId(1);
    let c = IndexId(2);
    let d = IndexId(3);
    let e = IndexId(4);

    let ext = vec![
        Index { id: a, range: occ },
        Index { id: b, range: occ },
    ];
    let term = Term {
        coeff: Ratio::from_integer(1),
        sum_indices: vec![
            Index { id: c, range: occ },
            Index { id: d, range: virt },
            Index { id: e, range: virt },
        ],
        factors: vec![
            Factor { tensor: TensorId(0), indices: vec![a, c] },
            Factor { tensor: TensorId(1), indices: vec![c, d] },
            Factor { tensor: TensorId(2), indices: vec![d, e] },
            Factor { tensor: TensorId(3), indices: vec![e, b] },
        ],
    };

    let result = parenthesize(&term, &ext, comp.ranges());

    let full = &result.memoir[&(0b1111u64, 0)];
    assert_eq!(full.evals.len(), 7);

    assert!(full.best_cost > 0);
    assert!(full.best_cost < u64::MAX);
}
