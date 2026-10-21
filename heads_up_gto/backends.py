"""Keep the optional neural dependencies out of the tabular application."""
from .solver import Solver


def create_solver(root, ranges, settings):
    if settings.backend == "tabular":
        return Solver(root, ranges, settings)
    if root.street not in (1, 2):
        raise ValueError("This Deep CFR pilot starts on the turn or river. Start a tabular flop solve, "
                         "navigate to a turn decision, then select Deep CFR and Refine decision.")
    try:
        from .deep_cfr import DeepCFRSolver
    except ImportError as exc:
        raise ValueError("Deep CFR needs optional dependencies. In the project folder run: "
                         ".venv/bin/python -m pip install '.[deep]'") from exc
    return DeepCFRSolver(root, ranges, settings)
