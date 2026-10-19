"""Cards, weighted combo ranges, and a direct 5–7 card evaluator."""
from collections import Counter
from functools import lru_cache
from itertools import combinations
import math
import re

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(range(52))
COMBOS = tuple(combinations(DECK, 2))


def card_text(card):
    return RANKS[card // 4] + SUITS[card % 4]


def parse_cards(text, count=None):
    if not isinstance(text, str):
        raise ValueError("Cards must be text, such as As Kd 7c.")
    text = text.strip().replace("10", "T")
    for symbol, suit in zip("♣♦♥♠", SUITS):
        text = text.replace(symbol, suit)
    clean = re.sub(r"[\s,]+", "", text)
    parts = re.findall(r"[2-9TJQKA][cdhs]", clean, re.I)
    if "".join(parts).lower() != clean.lower():
        raise ValueError("Use cards such as As Kd 7c (ranks 2–A; suits c, d, h, s).")
    cards = tuple(RANKS.index(p[0].upper()) * 4 + SUITS.index(p[1].lower()) for p in parts)
    if len(set(cards)) != len(cards):
        raise ValueError("A card cannot appear twice.")
    if count is not None and len(cards) != count:
        raise ValueError(f"Enter exactly {count} cards.")
    return cards


def combo_text(hand):
    return "".join(card_text(c) for c in sorted(hand, reverse=True))


def hand_class(hand):
    a, b = sorted(hand, reverse=True)
    ra, rb = a // 4, b // 4
    return RANKS[ra] + RANKS[rb] + ("" if ra == rb else "s" if a % 4 == b % 4 else "o")


def expand_hand(token):
    token = token.strip()
    if token == "*":
        return COMBOS
    if len(token) == 4:
        return (tuple(sorted(parse_cards(token, 2))),)
    match = re.fullmatch(r"([2-9TJQKA])([2-9TJQKA])([so]?)", token, re.I)
    if not match:
        raise ValueError(f"Invalid hand {token!r}. Use AA, AKs, AKo, AK, or AsKh; expand '+' and dash ranges first.")
    a, b = RANKS.index(match[1].upper()), RANKS.index(match[2].upper())
    suffix = match[3].lower()
    if a == b and suffix:
        raise ValueError("Pairs do not have a suited/offsuit suffix.")
    result = []
    for s in range(4):
        for t in range(4):
            if a == b and s >= t:
                continue
            if suffix == "s" and s != t or suffix == "o" and s == t:
                continue
            result.append(tuple(sorted((4 * a + s, 4 * b + t))))
    return tuple(result)


def parse_range(value):
    """Frequencies in [0,1]; later exact combos override hand-class entries."""
    if isinstance(value, str):
        items = []
        for token in value.split(","):
            if not token.strip():
                continue
            parts = token.strip().split(":")
            if len(parts) > 2:
                raise ValueError("Range weights use HAND:frequency, such as AKs:0.5.")
            items.append((parts[0], float(parts[1]) if len(parts) == 2 else 1.0))
    elif isinstance(value, dict):
        items = list(value.items())
    else:
        raise ValueError("A range must be comma-separated text or a hand-to-frequency object.")
    result = {}
    for token, weight in items:
        weight = float(weight)
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Every hand frequency must be between 0 and 1.")
        for hand in expand_hand(token):
            result[hand] = weight
    return {h: w for h, w in result.items() if w > 0}


def remove_blocked(weights, board):
    blocked = set(board)
    return {h: w for h, w in weights.items() if not blocked.intersection(h) and w > 0}


def _straight(ranks):
    bits = set(ranks)
    if 14 in bits:
        bits.add(1)
    for high in range(14, 5 - 1, -1):
        if all(r in bits for r in range(high - 4, high + 1)):
            return high
    return 0


@lru_cache(maxsize=200_000)
def evaluate(cards):
    """Comparable tuple; larger is stronger. Input contains 5, 6, or 7 cards."""
    ranks = [c // 4 + 2 for c in cards]
    counts = Counter(ranks)
    suits = [[] for _ in range(4)]
    for c in cards:
        suits[c % 4].append(c // 4 + 2)
    flush = next((sorted(s, reverse=True) for s in suits if len(s) >= 5), None)
    if flush and (straight_flush := _straight(flush)):
        return (8, straight_flush)
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    if groups[0][0] == 4:
        return (7, groups[0][1], max(r for r in ranks if r != groups[0][1]))
    trips = sorted((r for r, n in counts.items() if n >= 3), reverse=True)
    if trips:
        pairs = [r for r, n in counts.items() if n >= 2 and r != trips[0]]
        if pairs:
            return (6, trips[0], max(pairs))
    if flush:
        return (5, *flush[:5])
    if straight := _straight(ranks):
        return (4, straight)
    if trips:
        return (3, trips[0], *sorted((r for r in counts if r != trips[0]), reverse=True)[:2])
    pairs = sorted((r for r, n in counts.items() if n >= 2), reverse=True)
    if len(pairs) >= 2:
        return (2, *pairs[:2], max(r for r in counts if r not in pairs[:2]))
    if pairs:
        return (1, pairs[0], *sorted((r for r in counts if r != pairs[0]), reverse=True)[:3])
    return (0, *sorted(counts, reverse=True)[:5])


def score(hand, board):
    return evaluate(tuple(sorted((*hand, *board))))
