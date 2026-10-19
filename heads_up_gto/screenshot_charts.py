"""User-supplied GTO Wizard screenshots, manually approximated to 10% steps.

Each matrix is a standard 13x13 grid in AKQJT98765432 order. Above the
main diagonal is suited, below is offsuit. Codes: f=fold, c=call,
r=the smaller raise displayed at that node, j=all-in to 100 BB.
A bare letter means 100%; r3c7 means 30% raise / 70% call.
'-' means no visible strategy (hand absent at this node), NOT a fold.

Only the colored part of a cell determines its conditional action mix.
Black vertical space in Breakdown view encodes earlier reach, so is not
counted again. Frequencies within each visible cell sum to exactly 100%.
No screenshot or image-processing package is required at runtime.
"""
import re
import math

RANK_ORDER = "AKQJT98765432"
BB_FIVE_BET_HISTORY = "raise:2.5/raise:11.5/raise:26"
SB_FOUR_BET_HISTORY = "raise:2.5/raise:11.5"
ALLOWED_JAM_HISTORIES = (SB_FOUR_BET_HISTORY, BB_FIVE_BET_HISTORY)

# Screenshot 1, 19:26:33: SB opening. KJs is a darker hover/selection shade,
# not the all-in red: it remains fully present after Raise 2.5 in image 3.
OPEN = """
r r r r r r r r r r r r r
r r r r r r r r r r r r r
r r r r r r r r r r r r r
r r r r r r r r r r r r r
r r r r r r r r r r r r r
r r r r r r r r r r r r r
r r r r r r r r r r r r r
r r r r r r r r r r r r r
r r r r r r r r r r r r r
r r r r r r r r r r r r r
r r r r r2f8 f f r6f4 r r r r r
r r r f f f f f f r f r r
r r r6f4 f f f f f f f f f r
"""

# Screenshot 2, 19:26:40: BB facing SB Raise 2.5. r = Raise 11.5.
VS_OPEN = """
r r r r r r r3c7 c c r r9c1 r5c5 r2c8
r r r r r r1c9 c r1c9 r3c7 c r1c9 r1c9 c
r r r r r9c1 c c c c r1c9 c c c
r r3c7 r1c9 r r r3c7 r2c8 r5c5 c c c c c
r8c2 r2c8 r2c8 c r r7c3 r8c2 r6c4 r6c4 r1c9 c c c
r2c8 r3c7 r3c7 r2c8 r2c8 r r r9c1 r r4c6 c r1c9 c
c r3c7 r3c7 r2c8 r2c8 r2c8 r r r r3c7 r3c7 c c
r2c8 r2c8 r2c8 r1c9 r3c7 r1c9 r1c9 r6c4 r r r2c8 r1c9 c
r1c9 r2c8 r2c8 c4f6 c6f4 r1c9 r1c9 c r5c5 r r7c3 r3c7 c
c r1c9 c c1f9 f f f c c r1c9 r r5c5 c
r1c9 r1c9 c5f5 f f f f f c4f6 c c r3c7 c
r1c9 r1c9 f f f f f f f c2f8 f c r4c6
c c7f3 f f f f f f f f f f c
"""

# Screenshot 3, 19:27:04: SB facing BB Raise 11.5. r = Raise 26.
# Genuine dark-red segments are all-ins and remain separate from Raise 26.
VS_THREE_BET = """
r9c1 r c c c c c c c c c c c
j3r7 r c c c c c c c c r2c8 c r4c6
j1r5c4 c r c c c c c r2c8 r3c7 r5c5 r1c3f6 f
c r1c9 r2c8 r8c2 c c c c r2c3f5 r1f9 f f f
r2c8 j1r2c7 r3c7 j1r1c8 r6c4 c c c r1c4f5 f f f f
r3c7 f f r1f9 r1c6f3 j3r2c5 c c c f f f f
r3c1f6 f f f f f r1c9 c c c f f f
r1f9 f f f f f f c c c r1f9 f f
f f f f f f f f c c c f f
r3f7 f f f f f f f f c c c f
f f f f f - - f f f c c f
f f f - - - - - - f - c f
f f f - - - - - - - - - c
"""

# Screenshot 4, 19:27:11: BB facing SB Raise 26. Only jam/call/fold.
# Bar height is BB's earlier Raise 11.5 reach; conditional mixes use widths.
VS_FOUR_BET = """
c j j1c9 c c c c - - j7c3 j6c4 j7c3 c
j j j5c5 c c c c c c - c c -
j c j c c - - - - f - f -
c8f2 f f j c c f f - - - - -
f f f - j6c4 c f f f f - - -
f f f f f j5c5 c8f2 c1f9 f f - f -
f f f f f f j7c3 c c3f7 f f - -
f f f f f f f j8c2 c f f f -
f f f - - f f f j1c9 c c9f1 f -
f f f - - - - - - j3c7 c f -
f f - - - - - - - - - f -
f f - - - - - - - - - - f
f - - - - - - - - - - - -
"""

# Screenshot 5, 19:27:20: SB facing BB all-in 100 after Raise 26.
# All visible green/blue bars are pure call/fold, regardless of their height.
VS_FIVE_BET_JAM = """
c c - - - - - - - - - - -
c c - - - - - - - - f - f
c - c - - - - f f f f f -
- f f c - - - f f f - - -
f f f f c - - - f - - - -
f - - f f c - - - - - - -
f - - - - - c - - - - - -
f - - - - - - - - - f - -
- - - - - - - - - - - - -
f - - - - - - - - - - - -
- - - - - - - - - - - - -
- - - - - - - - - - - - -
- - - - - - - - - - - - -
"""

# Screenshot 2026-10-19 13:02:54: BB facing SB's 4-bet all-in to 100.
# Vertical bar height is BB's prior 3-bet reach. Only horizontal green/blue
# proportions determine the call/fold mix. Tiny mixes are rounded to tenths.
VS_FOUR_BET_JAM = """
c c c c c2f8 f f - - f f f f
c c c8f2 c1f9 f f f f f - f f -
c9f1 f c f f - - - - f - f -
f f f c f f f f - - - - -
f f f - c f f f f f - - -
f f f f f c f f f f - f -
f f f f f f c f f f f - -
f f f f f f f c f f f f -
f f f - - f f f c f f f -
f f f - - - - - - c9f1 f f -
f f - - - - - - - - - f -
f f - - - - - - - - - - f
f - - - - - - - - - - - -
"""

MATRICES = (
    ("", "SB open", "19:26:33", {"f": "fold", "r": "raise:2.5", "j": "raise:100"}, OPEN),
    ("raise:2.5", "BB versus 2.5 BB", "19:26:40", {"f": "fold", "c": "call", "r": "raise:11.5", "j": "raise:100"}, VS_OPEN),
    ("raise:2.5/raise:11.5", "SB versus 11.5 BB", "19:27:04", {"f": "fold", "c": "call", "r": "raise:26", "j": "raise:100"}, VS_THREE_BET),
    ("raise:2.5/raise:11.5/raise:26", "BB versus 26 BB", "19:27:11", {"f": "fold", "c": "call", "j": "raise:100"}, VS_FOUR_BET),
    ("raise:2.5/raise:11.5/raise:100", "BB versus SB 4-bet all-in", "2026-10-19 13:02:54", {"f": "fold", "c": "call"}, VS_FOUR_BET_JAM),
    ("raise:2.5/raise:11.5/raise:26/raise:100", "SB versus 100 BB all-in", "19:27:20", {"f": "fold", "c": "call"}, VS_FIVE_BET_JAM),
)


def _decode(matrix, actions):
    rows = [row.split() for row in matrix.strip().splitlines()]
    if len(rows) != 13 or any(len(row) != 13 for row in rows):
        raise ValueError("Screenshot matrices must contain 13 rows of 13 hands.")
    result = {action: {} for action in actions.values()}
    missing = []
    for i, row in enumerate(rows):
        for j, token in enumerate(row):
            hand = RANK_ORDER[min(i, j)] + RANK_ORDER[max(i, j)] + ("" if i == j else "s" if i < j else "o")
            if token == "-":
                missing.append(hand)
                continue
            parts = [(token, 10)] if token in actions else [(a, int(n)) for a, n in re.findall(r"([fcrj])(\d+)", token)]
            if token not in actions and "".join(f"{a}{n}" for a, n in parts) != token:
                raise ValueError(f"Invalid screenshot cell: {hand}={token}")
            if len({a for a, _ in parts}) != len(parts) or sum(n for _, n in parts) != 10:
                raise ValueError(f"Screenshot mix must total ten units: {hand}={token}")
            for code, units in parts:
                if code not in actions or not 1 <= units <= 10:
                    raise ValueError(f"Invalid action in screenshot cell: {hand}={token}")
                result[actions[code]][hand] = units / 10
    return result, missing


def screenshot_chart():
    nodes, absent, sources = {}, {}, []
    for history, label, timestamp, actions, matrix in MATRICES:
        nodes[history], absent[history] = _decode(matrix, actions)
        source = {"history": history, "label": label, "screenshot": timestamp if " " in timestamp else f"2026-10-18 {timestamp}"}
        if history == BB_FIVE_BET_HISTORY + "/raise:100":
            source["confirmed_by"] = "2026-10-19 13:03:03"
        sources.append(source)
    # Keep the screenshot transcription above intact. The active app tree only
    # permits SB's 4-bet and BB's 5-bet jams. Preserve the relative
    # non-jam mix and round back to tenths using largest remainders.
    adjusted = {}
    for history, actions in nodes.items():
        if history in ALLOWED_JAM_HISTORIES:
            continue
        removed = actions.pop("raise:100", {})
        for hand in removed:
            remaining = [action for action, weights in actions.items() if weights.get(hand, 0) > 0]
            total = sum(actions[action][hand] for action in remaining)
            if total <= 0:
                raise ValueError(f"No non-jam chart action remains for {hand} at {history!r}.")
            exact = [10 * actions[action][hand] / total for action in remaining]
            units = [math.floor(value + 1e-9) for value in exact]
            order = sorted(range(len(units)), key=lambda i: (-(exact[i] - units[i]), i))
            for i in order[:10 - sum(units)]:
                units[i] += 1
            for action, value in zip(remaining, units):
                if value:
                    actions[action][hand] = value / 10
                else:
                    actions[action].pop(hand)
        if removed:
            adjusted[history] = sorted(removed)
    return {
        "version": 1,
        "name": "GTO Wizard screenshots · rounded to 10%",
        "is_demo": False,
        "approximate": True,
        "frequency_step": 0.1,
        "game": {"stack_bb": 100, "small_blind_bb": .5, "big_blind_bb": 1, "ante_bb": 0, "rake": 0},
        "provenance": {
            "source": "Seven user-supplied screenshots covering six decision charts; Cash 100bb, Heads-up cEV",
            "method": "Visual estimates of conditional colored proportions, rounded in 10% steps with each visible cell normalized to 100%.",
            "reach": "Black space and colored bar height in Breakdown views indicate prior reach, not an additional fold probability. Prior reach is reconstructed from earlier nodes.",
            "caveat": "Approximate transcription, not an exact GTO Wizard export. Near rounding boundaries, either neighboring 10% step can be reasonable.",
            "kjs_open": "The darkened KJs cell in image 1 is treated as a selection shade for Raise 2.5: image 3 confirms full KJs reach after that action.",
            "all_in": "SB's 4-bet jam after 2.5 / 11.5 and BB's 5-bet jam after 2.5 / 11.5 / 26 are allowed. All-in 100 is raise:100 (total preflop contribution).",
            "jam_restriction": "Opening jams and BB 3-bet jams are disabled. The original SB 4-bet jam mixes are restored. Any removed early-jam frequency is redistributed over remaining actions in tenths. This restricted transcription is not a fresh equilibrium solve.",
            "adjusted_hands": adjusted,
            "coverage": "The displayed 2.5 / 11.5 / 26 tree plus SB 4-bet and BB 5-bet all-in responses; no limp or alternative-size strategies are inferred.",
            "screenshots": sources,
        },
        "unobserved_hands": absent,
        "nodes": nodes,
    }
