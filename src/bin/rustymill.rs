use std::path::PathBuf;

use rustymill::convert::{read_json, write_json};
use rustymill::cost::total_cost;
use rustymill::optimize::greedy_optimize;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let no_opt = args.iter().any(|a| a == "--no-opt");
    let positional: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with("--")).collect();

    if positional.is_empty() || positional.len() > 2 {
        eprintln!("Usage: {} [--no-opt] <input.json> [output.json]", args[0]);
        eprintln!("  Reads a TensorComputation from input.json,");
        eprintln!("  optimizes it with greedy factorization (unless --no-opt),");
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
        let n = greedy_optimize(&mut comp);

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
