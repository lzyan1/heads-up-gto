"""Exact best-response verification for small, exact-card river games only."""
from .cards import score


def river_exploitability(solver, max_pairs=400):
    if solver.root.street != 2 or solver.settings.mode != "exact":
        raise ValueError("Exact exploitability is only implemented for small exact-card river games.")
    pairs = []
    for a, wa in solver.ranges[0].items():
        for b, wb in solver.ranges[1].items():
            if not set(a).intersection(b):
                pairs.append((a, b, wa * wb))
                if len(pairs) > max_pairs:
                    raise ValueError("Too many hand pairs for the exact diagnostic.")
    total = sum(x[2] for x in pairs)
    weights = [x[2] / total for x in pairs]
    winners = []
    for a, b, _ in pairs:
        x, y = score(a, solver.root.board), score(b, solver.root.board)
        winners.append(0 if x > y else 1 if y > x else -1)

    def walk(state, reach, player, best_response):
        if state.terminal:
            return [state.payoff(state.winner if state.terminal == "fold" else w, player) for w in winners]
        actions = state.actions(solver.settings)
        policies = []
        for pair in pairs:
            policy, _, _ = solver.policy(state, pair[state.actor])
            policies.append(policy if policy is not None else [1 / len(actions)] * len(actions))
        if state.actor == player and best_response:
            children = [walk(state.apply(a), reach, player, True) for a in actions]
            values = {}
            for i, pair in enumerate(pairs):
                v = values.setdefault(pair[player], [0.0] * len(actions))
                for j in range(len(actions)):
                    v[j] += reach[i] * children[j][i]
            choice = {h: max(range(len(actions)), key=lambda j: v[j]) for h, v in values.items()}
            return [children[choice[pair[player]]][i] for i, pair in enumerate(pairs)]
        children = [walk(state.apply(a), [r * policy[j] for r, policy in zip(reach, policies)], player, best_response) for j, a in enumerate(actions)]
        return [sum(policy[j] * children[j][i] for j in range(len(actions))) for i, policy in enumerate(policies)]

    br_values = [sum(w * v for w, v in zip(weights, walk(solver.root, weights, p, True))) for p in (0, 1)]
    return {"nash_conv_bb": sum(br_values), "exploitability_bb": sum(br_values) / 2, "best_response_values_bb": br_values, "hand_pairs": len(pairs)}


# A descriptive sample threshold, not a confidence interval or convergence test.
SAMPLE_THRESHOLD = 200


def sample_summary(rows):
    """Summarize both counters, including missing hands when supplied by inspect."""
    counts = {"unseen": 0, "average_only": 0, "regret_only": 0,
              "low_samples": 0, "sampled": 0}
    masses = dict.fromkeys(counts, 0.0)
    from collections import Counter
    histograms = {key: Counter() for key in ("updates", "averages")}
    count = 0
    for row in rows:
        count += 1
        for key, histogram in histograms.items():
            histogram[row[key]] += 1
        u, a = row["updates"], row["averages"]
        category = ("unseen" if not u and not a else "average_only" if not u else
                    "regret_only" if not a else "low_samples" if min(u, a) < SAMPLE_THRESHOLD else "sampled")
        counts[category] += 1
        masses[category] += row.get("mass", 0.0)
    def distribution(key):
        histogram = histograms[key]
        values = sorted(histogram)
        def at(index):
            cumulative = 0
            for value in values:
                cumulative += histogram[value]
                if cumulative > index:
                    return value
            return 0
        return {"min": values[0] if values else 0,
                "median": (at((count-1)//2) + at(count//2))/2 if count else 0,
                "p90": at(int(.9*(count-1))) if count else 0,
                "max": values[-1] if values else 0,
                "total": sum(value * frequency for value, frequency in histogram.items())}
    return {"hands": count, "counts": counts, "mass": masses,
            "regret_updates": distribution("updates"), "average_samples": distribution("averages"),
            "sample_threshold": SAMPLE_THRESHOLD,
            "notice": "Sample counts are not a convergence or exploitability certificate."}
