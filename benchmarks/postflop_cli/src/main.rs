use postflop_solver::*;
use serde_json::{json, Value};
use std::io::Read;
use std::time::Instant;

fn range_from(v: &Value) -> Range {
    let mut hands = Vec::new();
    let mut weights = Vec::new();
    for entry in v.as_array().unwrap() {
        let text = entry[0].as_str().unwrap(); // e.g. "AsKd"
        hands.push((card_from_str(&text[0..2]).unwrap(), card_from_str(&text[2..4]).unwrap()));
        weights.push(entry[1].as_f64().unwrap() as f32);
    }
    Range::from_hands_weights(&hands, &weights).unwrap()
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let spot: Value = serde_json::from_str(&input).unwrap();

    if let Some(n) = spot["threads"].as_u64() {
        rayon::ThreadPoolBuilder::new().num_threads(n as usize).build_global().ok();
    }

    let board: Vec<&str> = spot["board"].as_str().unwrap().split_whitespace().collect();
    let card_config = CardConfig {
        range: [range_from(&spot["oop"]), range_from(&spot["ip"])],
        flop: flop_from_str(&board[..3].concat()).unwrap(),
        turn: card_from_str(board[3]).unwrap(),
        river: card_from_str(board[4]).unwrap(),
    };
    let bets = BetSizeOptions::try_from((
        spot["bets"].as_str().unwrap(),
        spot["raises"].as_str().unwrap(),
    ))
    .unwrap();
    let pot = spot["pot"].as_i64().unwrap() as i32;
    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: pot,
        effective_stack: spot["stack"].as_i64().unwrap() as i32,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bets.clone(), bets.clone()],
        turn_bet_sizes: [bets.clone(), bets.clone()],
        river_bet_sizes: [bets.clone(), bets],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: spot["add_allin_threshold"].as_f64().unwrap(),
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
    };
    let mut game = PostFlopGame::with_config(card_config, ActionTree::new(tree_config).unwrap()).unwrap();
    game.allocate_memory(false);

    let max_iter = spot["max_iterations"].as_u64().unwrap() as u32;
    let max_seconds = spot["max_seconds"].as_f64().unwrap();
    let target = spot["target_exploitability"].as_f64().unwrap() as f32 * pot as f32;
    let every = spot["check_every"].as_u64().unwrap_or(10) as u32;

    // Exploitability checks are excluded from the solve clock.
    let mut solve_seconds = 0.0f64;
    let mut history = Vec::new();
    let mut last = f32::INFINITY;
    for i in 0..max_iter {
        let t = Instant::now();
        solve_step(&game, i);
        solve_seconds += t.elapsed().as_secs_f64();
        if (i + 1) % every == 0 || i + 1 == max_iter || solve_seconds > max_seconds {
            last = compute_exploitability(&game);
            history.push(json!([i + 1, solve_seconds, last]));
            if last <= target || solve_seconds > max_seconds {
                break;
            }
        }
    }
    finalize(&mut game);
    game.cache_normalized_weights();
    let ev: Vec<f64> = (0..2)
        .map(|p| compute_average(&game.expected_values(p), game.normalized_weights(p)) as f64)
        .collect();
    println!(
        "{}",
        json!({
            "history": history,
            "exploitability": last,
            "solve_seconds": solve_seconds,
            "avg_ev": ev,
            "num_hands": [game.private_cards(0).len(), game.private_cards(1).len()],
            "memory_bytes": game.memory_usage().0,
            "threads": rayon::current_num_threads(),
        })
    );
}
