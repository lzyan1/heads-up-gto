"""Keep the optional neural dependencies out of the tabular application."""
from .solver import Solver


def create_solver(root, ranges, settings):
    if settings.backend == "tabular":
        return Solver(root, ranges, settings)
    validate_neural_root(root)
    try:
        from .deep_cfr import DeepCFRSolver
    except ImportError as exc:
        raise ValueError("Deep CFR needs optional dependencies. In the project folder run: "
                         ".venv/bin/python -m pip install '.[deep]'") from exc
    return DeepCFRSolver(root, ranges, settings)


def validate_neural_root(root):
    if (root.street not in (0, 1, 2) or len(root.board) != root.street + 3
            or root.pending or root.terminal):
        raise ValueError("Deep CFR starts at an active flop, turn or river decision.")
