use aethel_memory_rust::{run_jsonl, MemoryService};
#[cfg(unix)]
use aethel_memory_rust::run_unix_socket;
use std::io::{self, BufReader};
use std::path::PathBuf;

fn arg_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|value| value == name).and_then(|index| args.get(index + 1).cloned())
}

fn main() -> Result<(), String> {
    let dimension = arg_value("--dimension").unwrap_or_else(|| "384".into()).parse().map_err(|_| "--dimension debe ser entero")?;
    let capacity = arg_value("--capacity").unwrap_or_else(|| "2048".into()).parse().map_err(|_| "--capacity debe ser entero")?;
    let snapshot = arg_value("--snapshot").map(PathBuf::from);
    let mut service = MemoryService::from_snapshot(dimension, capacity, snapshot)?;
    #[cfg(unix)]
    if let Some(socket) = arg_value("--socket") {
        eprintln!("Aethel Rust Memory socket ready at {socket}");
        return run_unix_socket(&mut service, &PathBuf::from(socket));
    }
    eprintln!("Aethel Rust Memory JSONL ready; use {{\"op\":\"health\"}} on stdin.");
    run_jsonl(&mut service, BufReader::new(io::stdin().lock()), io::stdout().lock())
}
