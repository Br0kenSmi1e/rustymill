#!/usr/bin/env python3
"""End-to-end test: gristmill -> JSON -> rustymill optimize -> JSON -> gristmill."""

import sys
import os
import json
import subprocess
import tempfile

# Add gristmill to path
sys.path.insert(0, os.path.expanduser('~/pycode/gristmill'))
sys.path.insert(0, os.path.expanduser('~/pycode/drudge'))

from sympy import IndexedBase, Symbol, Rational
from drudge import Range
from dummy_spark import SparkContext  # or use a real SparkContext

# Try to set up a minimal drudge
try:
    from pyspark import SparkContext
    ctx = SparkContext('local[1]', 'test')
except Exception:
    # Fallback: try without spark
    ctx = None

from drudge import Drudge

if ctx is None:
    print("Warning: No SparkContext available, trying without it")
    # We need a drudge instance — let's see if we can create one
    sys.exit(1)

dr = Drudge(ctx)

# Define ranges
occ = Range('occ', 0, 10)
virt = Range('virt', 0, 100)

# Set up dummies
a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
dr.set_dumms(occ, [a, b, c])

i, j = Symbol('i'), Symbol('j')
dr.set_dumms(virt, [i, j])

# Define tensors
X = IndexedBase('X')
Y = IndexedBase('Y')
U = IndexedBase('U')
V = IndexedBase('V')
t = IndexedBase('t')

# Build: t[a,b] = 4*X[a,c]*U[c,b] + 2*X[a,c]*V[c,b] - 2*Y[a,c]*U[c,b] - Y[a,c]*V[c,b]
from drudge import Term
terms = [
    Term(((c, occ),), Rational(4) * X[a, c] * U[c, b], ()),
    Term(((c, occ),), Rational(2) * X[a, c] * V[c, b], ()),
    Term(((c, occ),), Rational(-2) * Y[a, c] * U[c, b], ()),
    Term(((c, occ),), Rational(-1) * Y[a, c] * V[c, b], ()),
]

tensor_def = dr.define(t, (a, occ), (b, occ), terms=terms)
print("Original:")
print(tensor_def)

# Export to JSON
from gristmill.json_io import RustyMillConverter
converter = RustyMillConverter(dr)
json_str = converter.export_json([tensor_def])
print("\nExported JSON:")
print(json_str)

# Write to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    f.write(json_str)
    input_path = f.name

# Run rustymill optimizer
rustymill_bin = os.path.expanduser('~/rcode/rustymill/target/debug/rustymill')
output_path = input_path.replace('.json', '_opt.json')

result = subprocess.run(
    [rustymill_bin, input_path, output_path],
    capture_output=True, text=True
)
print("\nRustymill output:")
print(result.stderr)

if result.returncode != 0:
    print("Error:", result.stderr)
    sys.exit(1)

# Read optimized JSON
with open(output_path) as f:
    opt_json = f.read()
print("Optimized JSON:")
print(opt_json)

# Import back to gristmill
optimized = converter.import_json(opt_json)
print("\nOptimized TensorDefs:")
for td in optimized:
    print(td)

# Cleanup
os.unlink(input_path)
os.unlink(output_path)

if ctx:
    ctx.stop()
