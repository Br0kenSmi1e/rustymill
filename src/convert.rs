//! JSON conversion utilities for TensorComputation.
//!
//! Provides functions to read/write TensorComputation from/to JSON files,
//! enabling interop with gristmill's Python-side RustyMillConverter.

use std::fs;
use std::path::Path;

use crate::repr::TensorComputation;

/// Read a TensorComputation from a JSON file.
pub fn read_json(path: &Path) -> Result<TensorComputation, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse JSON from {}: {}", path.display(), e))
}

/// Write a TensorComputation to a JSON file.
pub fn write_json(comp: &TensorComputation, path: &Path) -> Result<(), String> {
    let json = serde_json::to_string_pretty(comp)
        .map_err(|e| format!("Failed to serialize: {}", e))?;
    fs::write(path, json)
        .map_err(|e| format!("Failed to write {}: {}", path.display(), e))
}

/// Parse a TensorComputation from a JSON string.
pub fn from_json(json: &str) -> Result<TensorComputation, String> {
    serde_json::from_str(json)
        .map_err(|e| format!("Failed to parse JSON: {}", e))
}

/// Serialize a TensorComputation to a JSON string.
pub fn to_json(comp: &TensorComputation) -> Result<String, String> {
    serde_json::to_string_pretty(comp)
        .map_err(|e| format!("Failed to serialize: {}", e))
}
