use std::path::PathBuf;

use rustymill::convert::{read_json, write_json};
use rustymill::cost::total_cost;
use rustymill::optimize::{greedy_optimize, mcts_optimize};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let no_opt = args.iter().any(|a| a == "--no-opt");
    let mcts_pos = args.iter().position(|a| a == "--mcts");
    let mcts_iters: Option<u32> = mcts_pos
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    // Collect positional args: skip program name, flags, and --mcts's value
    let mut skip_next = false;
    let mut positional: Vec<&String> = Vec::new();
    for (i, a) in args.iter().enumerate().skip(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a == "--mcts" {
            skip_next = true;
            continue;
        }
        if a.starts_with("--") {
            continue;
        }
        positional.push(a);
    }

    if positional.is_empty() || positional.len() > 2 {
        eprintln!("Usage: {} [--no-opt] [--mcts <iterations>] <input.json> [output.json]", args[0]);
        eprintln!("  Reads a TensorComputation from input.json,");
        eprintln!("  optimizes it with greedy factorization (unless --no-opt),");
        eprintln!("  optionally uses MCTS with given iterations (--mcts N),");
        eprintln!("  and writes the result to output.json (or stdout).");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(positional[0]);
    let mut comp = read_json(&input_path).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let cost_before = total_cost(&comp);
    eprintln!("Input: {} definitions, cost = {}", comp.definitions().len(), cost_before);

    if !no_opt {
        let n = if let Some(iters) = mcts_iters {
            eprintln!("Running MCTS with {} iterations...", iters);
            mcts_optimize(&mut comp, iters, std::f64::consts::SQRT_2);
            greedy_optimize(&mut comp)
        } else {
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
    } else {
        eprintln!("Skipping optimization (--no-opt)");
    }

    if positional.len() == 2 {
        let output_path = PathBuf::from(positional[1]);
        write_json(&comp, &output_path).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });
        eprintln!("Written to {}", output_path.display());
    } else {
        let json = rustymill::to_json(&comp).unwrap();
        println!("{}", json);
    }
}
