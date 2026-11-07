"""Full-width CFR solvers: the whole postflop game (flop, turn, river) and river-only variants.

``PostflopCFR`` (solver.py) is the main solver with a numpy reference engine and a compiled numba
engine (kernels.py); ``FullTreeSolver`` (full_tree.py) adapts it to the study app. ``CFRPlusRiver``
and ``BatchedRiver`` are the river-only predecessors, kept for the dataset generator and as checks.
"""
from .batched_river import BatchedRiver
from .full_tree import FullTreeSolver
from .single_river import CFRPlusRiver
from .solver import PostflopCFR

__all__ = ['BatchedRiver', 'CFRPlusRiver', 'FullTreeSolver', 'PostflopCFR']
