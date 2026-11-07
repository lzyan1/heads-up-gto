"""Deterministic hand-category bots used only to generate river inputs."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from ..cards import score

BOT_VERSION = 'face-value-v1'


@lru_cache(maxsize=32768)
def hand_features(hand, board):
    """Return made/draw strength, bluff suitability, and a readable category.

    No equity calculation, opponent cards, range knowledge, or future cards.
    Scores roughly order air < draws < pairs < two pair < trips < straights.
    These deliberately naive ranks are not estimates of expected winnings.
    """
    ranks = [c // 4 + 2 for c in (*hand, *board)]
    hole = [c // 4 + 2 for c in hand]
    public = sorted({c // 4 + 2 for c in board}, reverse=True)
    evaluated = score(hand, board)
    category = evaluated[0]
    if category >= 2:
        value = (0, 0, 9, 10, 11, 12, 13, 14, 15)[category]
        label = ('air', 'pair', 'two_pair', 'trips', 'straight', 'flush',
                 'full_house', 'quads', 'straight_flush')[category]
    elif category == 1 and evaluated[1] in hole:
        pair = evaluated[1]
        if hole[0] == hole[1] and pair > public[0]:
            value, label = 8, 'overpair'
        elif pair == public[0]:
            value, label = 7, 'top_pair'
        elif len(public) > 1 and pair >= public[1]:
            value, label = 6, 'middle_pair'
        else:
            value, label = 5, 'weak_pair'
    else:
        value, label = (1, 'overcards') if min(hole) > public[0] else (0, 'air')
    # High cards break category ties without changing the category ordering.
    value += max(hole) / 100
    draws = max(hole) / 70  # A small fraction of air hands can bluff.
    if len(board) < 5 and category < 4:
        distinct = set(ranks)
        if 14 in distinct:
            distinct.add(1)
        outs = set()
        for rank in range(2, 15):
            extended = distinct | {rank} | ({1} if rank == 14 else set())
            if any(set(range(low, low + 5)) <= extended for low in range(1, 11)):
                outs.add(rank)
        straight_draw = 3 if len(outs) >= 2 else 2 if outs else 0
        suits = Counter(c % 4 for c in (*hand, *board))
        flush_draw = any(n == 4 and any(c % 4 == suit for c in hand)
                         for suit, n in suits.items())
        if flush_draw:
            draw, draw_label = (5, 'combo_draw') if straight_draw else (4, 'flush_draw')
        else:
            draw, draw_label = straight_draw, 'open_ended' if straight_draw == 3 else 'gutshot'
        draws = max(draws, draw)
        if draw > value:
            value, label = draw + max(hole) / 100, draw_label
    return value, draws, label


@dataclass(frozen=True)
class NaiveBot:
    value_threshold: float
    bluff_threshold: float

    @classmethod
    def sample(cls, rng):
        # From weak/middle pair value bettors to bots waiting for two pair.
        # Bluff cutoffs span air through strong combo draws and no bluff at all.
        return cls(rng.uniform(5, 9), rng.uniform(0, 6))

    def action(self, state, hand, actions):
        strength, bluff_score, _ = hand_features(tuple(sorted(hand)), state.board)
        to_call = max(state.paid) - state.paid[state.actor]
        aggressive = next((a for a in actions if a.kind in ('bet', 'raise')), None)
        value = strength >= self.value_threshold + (1 if to_call else 0)
        bluff = strength < 6 and bluff_score >= self.bluff_threshold
        if aggressive is not None and (value or bluff):
            return aggressive
        if to_call:
            price = to_call / (state.pot + to_call)
            # Calling uses the same value threshold, relaxed by three category
            # points and tightened for expensive calls. No third sampled knob.
            cutoff = self.value_threshold - 3 + 2 * price
            return next(a for a in actions if a.kind == ('call' if strength >= cutoff else 'fold'))
        return next(a for a in actions if a.kind == 'check')
