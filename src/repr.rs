use num::rational::Ratio;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// ID newtypes
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RangeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IndexId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TensorId(pub u32);

// ---------------------------------------------------------------------------
// Symmetry
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymAction {
    Identity,
    Negate,
    Conjugate,
    NegateConjugate,
}

impl SymAction {
    pub fn combine(self, other: SymAction) -> SymAction {
        let (sn, sc) = self.to_bits();
        let (on, oc) = other.to_bits();
        Self::from_bits(sn ^ on, sc ^ oc)
    }

    fn to_bits(self) -> (bool, bool) {
        match self {
            SymAction::Identity => (false, false),
            SymAction::Negate => (true, false),
            SymAction::Conjugate => (false, true),
            SymAction::NegateConjugate => (true, true),
        }
    }

    fn from_bits(negate: bool, conjugate: bool) -> SymAction {
        match (negate, conjugate) {
            (false, false) => SymAction::Identity,
            (true, false) => SymAction::Negate,
            (false, true) => SymAction::Conjugate,
            (true, true) => SymAction::NegateConjugate,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymGenerator {
    pub perm: Vec<usize>,
    pub action: SymAction,
}

impl SymGenerator {
    pub fn apply<T: Copy>(&self, indices: &[T]) -> (Vec<T>, SymAction) {
        assert_eq!(self.perm.len(), indices.len());
        let permuted = self.perm.iter().map(|&p| indices[p]).collect();
        (permuted, self.action)
    }
}

// ---------------------------------------------------------------------------
// Tensor data structures
// ---------------------------------------------------------------------------

pub type Rational = Ratio<i64>;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Range {
    pub id: RangeId,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Index {
    pub id: IndexId,
    pub range: RangeId,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    pub id: TensorId,
    pub symmetry: Vec<SymGenerator>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Factor {
    pub tensor: TensorId,
    pub indices: Vec<IndexId>,
}

impl fmt::Display for Factor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tensor.0)?;
        write!(f, "[")?;
        for (i, ix) in self.indices.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", ix.0)?;
        }
        write!(f, "]")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Term {
    pub coeff: Rational,
    pub sum_indices: Vec<Index>,
    pub factors: Vec<Factor>,
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let one = Rational::new(1, 1);
        if !self.sum_indices.is_empty() {
            write!(f, "Σ[")?;
            for (i, ix) in self.sum_indices.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", ix.id.0)?;
            }
            write!(f, "] ")?;
        }
        if self.coeff != one {
            write!(f, "{}", self.coeff)?;
            if !self.factors.is_empty() {
                write!(f, " * ")?;
            }
        }
        if self.factors.is_empty() {
            if self.coeff == one && self.sum_indices.is_empty() {
                write!(f, "1")?;
            }
        } else {
            for (i, fac) in self.factors.iter().enumerate() {
                if i > 0 {
                    write!(f, " * ")?;
                }
                write!(f, "{fac}")?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDef {
    pub base: TensorId,
    pub ext_indices: Vec<Index>,
    pub terms: Vec<Term>,
}

impl fmt::Display for TensorDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base.0)?;
        write!(f, "[")?;
        for (i, ix) in self.ext_indices.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", ix.id.0)?;
        }
        write!(f, "] = ")?;
        if self.terms.is_empty() {
            write!(f, "0")?;
        } else {
            for (i, t) in self.terms.iter().enumerate() {
                if i > 0 {
                    write!(f, " + ")?;
                }
                write!(f, "{t}")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TensorComputation container and builder API
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorComputation {
    ranges: Vec<Range>,
    tensors: Vec<TensorInfo>,
    definitions: Vec<TensorDef>,
}

impl TensorComputation {
    pub fn new() -> Self {
        TensorComputation {
            ranges: Vec::new(),
            tensors: Vec::new(),
            definitions: Vec::new(),
        }
    }

    pub fn add_range(&mut self, size: u64) -> RangeId {
        let id = RangeId(self.ranges.len() as u32);
        self.ranges.push(Range { id, size });
        id
    }

    pub fn add_tensor(&mut self, symmetry: Vec<SymGenerator>) -> TensorId {
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorInfo {
            id,
            symmetry,
        });
        id
    }

    pub fn add_definition(
        &mut self,
        base: TensorId,
        ext_indices: Vec<Index>,
        terms: Vec<Term>,
    ) {
        self.definitions.push(TensorDef {
            base,
            ext_indices,
            terms,
        });
    }

    pub fn ranges(&self) -> &[Range] {
        &self.ranges
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn definitions(&self) -> &[TensorDef] {
        &self.definitions
    }

    pub fn definitions_mut(&mut self) -> &mut Vec<TensorDef> {
        &mut self.definitions
    }

    pub fn next_tensor_id(&self) -> TensorId {
        TensorId(self.tensors.len() as u32)
    }
}

impl fmt::Display for TensorComputation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for def in &self.definitions {
            if !first {
                write!(f, "\n")?;
            }
            first = false;
            write!(f, "{def}")?;
        }
        Ok(())
    }
}

impl Default for TensorComputation {
    fn default() -> Self {
        Self::new()
    }
}
