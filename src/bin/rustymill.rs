use std::path::PathBuf;

use rustymill::convert::{read_json, write_json};
use rustymill::cost::total_cost;
use rustymill::optimize::{greedy_optimize, mcts_optimize};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse --iter <N> option
    let iter_pos = args.iter().position(|a| a == "--iter");
    let mcts_iters: Option<u32> = iter_pos
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    // Collect positional args: skip program name, --iter flag and its value
    let mut skip_next = false;
    let mut positional: Vec<&String> = Vec::new();
    for a in args.iter().skip(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a == "--iter" {
            skip_next = true;
            continue;
        }
        if a.starts_with("--") {
            eprintln!("Error: unknown option '{}'", a);
            eprintln!("Usage: {} [--iter <N>] <input.json> [output.json]", args[0]);
            std::process::exit(1);
        }
        positional.push(a);
    }

    if positional.is_empty() || positional.len() > 2 {
        eprintln!("Usage: {} [--iter <N>] <input.json> [output.json]", args[0]);
        eprintln!("  Reads a TensorComputation from input.json,");
        eprintln!("  optimizes with MCTS+greedy (if --iter specified) or greedy-only,");
        eprintln!("  and writes result to output.json (or stdout if omitted).");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(positional[0]);
    let mut comp = read_json(&input_path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", input_path.display(), e);
        std::process::exit(1);
    });

    let cost_before = total_cost(&comp);
    eprintln!(
        "Input: {} definitions, cost = {}",
        comp.definitions().len(),
        cost_before
    );

    let n = if let Some(iters) = mcts_iters {
        eprintln!("Running MCTS with {} iterations...", iters);
        let n_mcts = mcts_optimize(&mut comp, iters, 1.414);
        let n_rollout = greedy_optimize(&mut comp);
        n_mcts + n_rollout
    } else {
        eprintln!("Running greedy optimization...");
        greedy_optimize(&mut comp)
    };

    let cost_after = total_cost(&comp);
    eprintln!(
        "Output: {} definitions, cost = {} (saving = {}, {} factorizations applied)",
        comp.definitions().len(),
        cost_after,
        cost_before as i64 - cost_after as i64,
        n,
    );

    if positional.len() == 2 {
        let output_path = PathBuf::from(positional[1]);
        write_json(&comp, &output_path).unwrap_or_else(|e| {
            eprintln!("Error writing {}: {}", output_path.display(), e);
            std::process::exit(1);
        });
        eprintln!("Written to {}", output_path.display());
    } else {
        let json = rustymill::to_json(&comp).unwrap();
        println!("{}", json);
    }
}