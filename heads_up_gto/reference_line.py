"""Read-only comparison with the supplied screenshot, using the existing tree.

The screenshot contains a selected line, not strategy frequencies. Its runout
is used only for inspection; it never constrains sampling or changes bet sizes.
"""
from .cards import parse_cards

FLOP = parse_cards("As Kd 7c")
REFERENCE = (
    ("Flop · BB", "check", 5, (97.5, 97.5), ("check", "bet:1.7")),
    ("Flop · SB after check", "bet:1.7", 5, (97.5, 97.5), ("check", "bet:1.7")),
    ("Flop · BB facing bet", "call", 6.7, (97.5, 95.8), ("fold", "call", "raise:9.6")),
    ("Turn · BB", "check", 8.4, (95.8, 95.8), ("check", "bet:16.3")),
    ("Turn · SB after check", "bet:16.3", 8.4, (95.8, 95.8), ("check", "bet:16.3")),
    ("Turn · BB facing bet", "call", 24.7, (95.8, 79.5), ("fold", "call", "raise:41.1")),
    ("River · BB", None, 41, (79.5, 79.5), None),
)


def supports_reference(root):
    return (root.street == 0 and set(root.board) == set(FLOP)
            and root.base_pot == 5 and root.start_stacks == (97.5, 97.5)
            and root.invested == (0, 0) and root.paid == (0, 0)
            and root.actor == 0 and not root.checked and not root.path
            and not root.pending and not root.terminal
            and root.sizing_profile == "single_raised")


def comparison_report(solver):
    if not supports_reference(solver.root):
        raise ValueError("Use the original As Kd 7c flop solve after SB raises to 2.5 and BB calls.")
    state, path, nodes = solver.root, [], []
    for index, (label, selected, pot, stacks, menu) in enumerate(REFERENCE):
        view = solver.inspect(list(path))
        action = None
        if selected is not None:
            kind = selected.split(":")[0]
            # The app offers one scenario bet plus all-in. Follow that scenario
            # bet, never snap to the screenshot amount or create a new action.
            choices = [a for a in state.actions(solver.settings) if a.kind == kind]
            if not choices:
                raise ValueError(f"The current tree cannot follow the reference at {label}.")
            action = min(choices, key=lambda a: a.amount)
        nodes.append({"label": label, "model": view,
                      "model_selected": action.key if action else None,
                      "reference": {"pot": pot, "stacks": stacks, "selected": selected,
                                    "actions": menu, "probabilities": None},
                      "same_amounts": view["pot"] == pot and tuple(view["stacks"]) == stacks,
                      "same_menu": [a["key"] for a in view["actions"]] == list(menu) if menu else None,
                      "frequency_error": None})
        if action:
            path.append(action.key)
            state = state.apply(action)
        if index in (2, 5):
            card = "2h" if index == 2 else "3d"
            state = state.deal(parse_cards(card, 1)[0])
            path.append("deal:" + card)
    return {"version": 1, "title": "Supplied GTO Wizard line",
            "preflop": ["raise:2.5", "call"], "board": ["As", "Kd", "7c", "2h", "3d"],
            "reference_frequencies_available": False, "exact_benchmark": False,
            "notice": "Same board and action types, using this app's fixed scenario sizes. "
                      "Different sizes change pots and reaching ranges. The screenshot has no strategy "
                      "frequencies, so no frequency error or GTO accuracy score is available. "
                      "All model views use the same flop solve; no additional training or refinement is run. "
                      "Preflop chart weights are rounded to 10%; the reference's full tree and rake are unknown.",
            "stats": solver.stats(), "path": path, "nodes": nodes}
