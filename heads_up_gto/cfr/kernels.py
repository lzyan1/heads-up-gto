"""numba kernels for PostflopCFR: the whole DCFR traversal in compiled, threaded code.

The Python side flattens the public tree into integer tables (``T`` below). A *block* is one
street's betting tree: decision, fold and showdown nodes in preorder, plus *chance* leaves
that continue into a child block with ``m`` times as many contexts. A traversal for one
context walks its block's nodes on length-N vectors that stay in cache:

    forward   reach at every node (opponent strategy fixed while the traverser is updated)
    leaves    fold / showdown values; chance leaves recurse into one child context per card
    backward  node values, and for the traverser: regret and strategy-sum updates

Contexts below the root block are independent (they own disjoint regret slices), so the root
evaluates its chance children in a ``prange`` and everything deeper runs serially per thread.

Modes: 0 train (opponent plays the current strategy, traverser updates); 1 best response to
the average strategy; 2 value of the average strategy for both players.

Table layout ``T``:
  0 ntype  1 actor  2 nact  3 children[g, a] (block-local)  4 reg_off  5 pay[g, player, 3]
  6 ch_blk  7 ch_m  8 ch_draw  9 ch_w  10 ch_cards (offset)  11 prep_off  12 blk_off  13 blk_n
  14 cards_flat  15 prep_flat  16 c1  17 c2  18 card offsets into the holder lists (53)
Regret / strategy buffers are flat, laid out ``[context][action][hand]`` per decision node.
Showdown prep per context is ``[order N | card x N | card y N | group end N]`` (hands in strength order).
"""
import numpy as np
from numba import njit, prange

DECISION, FOLD, SHOWDOWN, CHANCE = 0, 1, 2, 3


@njit(cache=True)
def _straight_high(mask):
    """High card of the best straight in a rank bitmask (bit r for rank r; ace also plays low), else 0."""
    if (mask >> 14) & 1:
        mask |= 2
    for high in range(14, 4, -1):
        if (mask >> (high - 4)) & 31 == 31:
            return high
    return 0


@njit(cache=True)
def _encode(cat, a, b, c, d, e):
    return (cat << 20) | (a << 16) | (b << 12) | (c << 8) | (d << 4) | e


@njit(cache=True)
def _score7(cs):
    """Hand strength of seven cards as one integer. Equal integers tie and larger beats smaller,
    exactly matching the tuples from ``cards.evaluate`` (category, then its tie-break ranks)."""
    cnt = np.zeros(15, np.int64)
    scnt = np.zeros(4, np.int64)
    smask = np.zeros(4, np.int64)
    mask = 0
    for i in range(7):
        r = cs[i] // 4 + 2
        s = cs[i] % 4
        cnt[r] += 1
        scnt[s] += 1
        smask[s] |= 1 << r
        mask |= 1 << r
    flush = -1
    for s in range(4):
        if scnt[s] >= 5:
            flush = s
    if flush >= 0:
        high = _straight_high(smask[flush])
        if high > 0:
            return _encode(8, high, 0, 0, 0, 0)
    quad = 0
    trip = 0
    trip2 = 0
    for r in range(14, 1, -1):
        if cnt[r] == 4:
            quad = r
        elif cnt[r] >= 3:
            if trip == 0:
                trip = r
            elif trip2 == 0:
                trip2 = r
    if quad > 0:
        kick = 0
        for r in range(14, 1, -1):
            if r != quad and cnt[r] > 0:
                kick = r
                break
        return _encode(7, quad, kick, 0, 0, 0)
    if trip > 0:
        pair = 0
        for r in range(14, 1, -1):
            if r != trip and cnt[r] >= 2:
                pair = r
                break
        if pair > 0:
            return _encode(6, trip, pair, 0, 0, 0)
    if flush >= 0:
        top = np.zeros(5, np.int64)
        k = 0
        for r in range(14, 1, -1):
            if (smask[flush] >> r) & 1 and k < 5:
                top[k] = r
                k += 1
        return _encode(5, top[0], top[1], top[2], top[3], top[4])
    high = _straight_high(mask)
    if high > 0:
        return _encode(4, high, 0, 0, 0, 0)
    other = np.zeros(5, np.int64)
    if trip > 0:
        k = 0
        for r in range(14, 1, -1):
            if r != trip and cnt[r] > 0 and k < 2:
                other[k] = r
                k += 1
        return _encode(3, trip, other[0], other[1], 0, 0)
    p0 = 0
    p1 = 0
    for r in range(14, 1, -1):
        if cnt[r] >= 2:
            if p0 == 0:
                p0 = r
            elif p1 == 0:
                p1 = r
    if p1 > 0:
        kick = 0
        for r in range(14, 1, -1):
            if r != p0 and r != p1 and cnt[r] > 0:
                kick = r
                break
        return _encode(2, p0, p1, kick, 0, 0)
    if p0 > 0:
        k = 0
        for r in range(14, 1, -1):
            if r != p0 and cnt[r] > 0 and k < 3:
                other[k] = r
                k += 1
        return _encode(1, p0, other[0], other[1], other[2], 0)
    k = 0
    for r in range(14, 1, -1):
        if cnt[r] > 0 and k < 5:
            other[k] = r
            k += 1
    return _encode(0, other[0], other[1], other[2], other[3], other[4])


@njit(cache=True, parallel=True)
def board_strengths(boards, hands):
    """Strength of every hand on every 5-card board: ``[boards, hands]`` int64.

    A hand that shares a card with the board gets a distinct negative value, so board-blocked
    hands (which carry no reach) sort first and never form tie groups.
    """
    K = boards.shape[0]
    N = hands.shape[0]
    out = np.empty((K, N), np.int64)
    for k in prange(K):
        cs = np.empty(7, np.int64)
        for i in range(5):
            cs[i] = boards[k, i]
        for h in range(N):
            cs[5] = hands[h, 0]
            cs[6] = hands[h, 1]
            dead = False
            for i in range(5):
                if boards[k, i] == cs[5] or boards[k, i] == cs[6]:
                    dead = True
            out[k, h] = -1 - h if dead else _score7(cs)
    return out


@njit(cache=True, fastmath=True)
def _sigma(buf, off, c, A, N, sig, tot):
    """Regret matching on the positive part of ``buf`` (uniform where nothing is positive).

    Two, three and four actions are fused into one pass over the hands (the common cases).
    """
    base = off + c * A * N
    if A == 2:
        for h in range(N):
            a0 = max(buf[base + h], 0.0)
            a1 = max(buf[base + N + h], 0.0)
            t = a0 + a1
            if t > 1e-12:
                inv = 1.0 / t
                sig[0, h] = a0 * inv
                sig[1, h] = a1 * inv
            else:
                sig[0, h] = 0.5
                sig[1, h] = 0.5
        return
    if A == 3:
        for h in range(N):
            a0 = max(buf[base + h], 0.0)
            a1 = max(buf[base + N + h], 0.0)
            a2 = max(buf[base + 2 * N + h], 0.0)
            t = a0 + a1 + a2
            if t > 1e-12:
                inv = 1.0 / t
                sig[0, h] = a0 * inv
                sig[1, h] = a1 * inv
                sig[2, h] = a2 * inv
            else:
                sig[0, h] = 1.0 / 3.0
                sig[1, h] = 1.0 / 3.0
                sig[2, h] = 1.0 / 3.0
        return
    if A == 4:
        for h in range(N):
            a0 = max(buf[base + h], 0.0)
            a1 = max(buf[base + N + h], 0.0)
            a2 = max(buf[base + 2 * N + h], 0.0)
            a3 = max(buf[base + 3 * N + h], 0.0)
            t = a0 + a1 + a2 + a3
            if t > 1e-12:
                inv = 1.0 / t
                sig[0, h] = a0 * inv
                sig[1, h] = a1 * inv
                sig[2, h] = a2 * inv
                sig[3, h] = a3 * inv
            else:
                sig[0, h] = 0.25
                sig[1, h] = 0.25
                sig[2, h] = 0.25
                sig[3, h] = 0.25
        return
    for h in range(N):
        tot[h] = 0.0
    for a in range(A):
        o = base + a * N
        for h in range(N):
            v = buf[o + h]
            if v < 0.0:
                v = 0.0
            sig[a, h] = v
            tot[h] += v
    uniform = 1.0 / A
    for h in range(N):
        if tot[h] > 1e-12:
            tot[h] = 1.0 / tot[h]
        else:
            tot[h] = -1.0
    for a in range(A):
        for h in range(N):
            t = tot[h]
            if t > 0.0:
                sig[a, h] = sig[a, h] * t
            else:
                sig[a, h] = uniform


@njit(cache=True, fastmath=True)
def _fold_leaf(reach, c1, c2, N, value, out, mass):
    """Fold payoff times the opponent reach compatible with each hand (no shared card).

    Hands arrive sorted by (c1, c2), so consecutive hands share ``c1``: that mass is summed in
    a register per run instead of read-modify-write on the same memory cell every hand.
    """
    for k in range(52):
        mass[k] = 0.0
    total = 0.0
    run = 0.0
    cur = c1[0]
    for h in range(N):
        r = reach[h]
        total += r
        x = c1[h]
        if x != cur:
            mass[cur] += run
            run = 0.0
            cur = x
        run += r
        mass[c2[h]] += r
    mass[cur] += run
    for h in range(N):
        out[h] = value * (total - mass[c1[h]] - mass[c2[h]] + reach[h])


@njit(cache=True, fastmath=True)
def _showdown_leaf(prep, base, reach, N, T, win_pay, loss_pay, tie_pay, out, winb, tieb, mass):
    """Win/loss/tie reach per own hand in one sweep over hands in strength order.

    ``cm[c]`` is the opponent reach already swept that holds card ``c`` and ``tot`` the total:
    a hand's weaker opponents with no shared card are ``tot - cm[x] - cm[y]``. Hands tied in
    strength are handled as a group (weaker mass, then add the group, then the tie mass, with
    the identical combo added back once since both of its cards subtract it). A hand alone in
    its group has no tie mass. Per context ``prep`` holds ``[order | card x | card y | group
    end]`` for the hands sorted by strength.
    """
    c1 = T[16]
    c2 = T[17]
    cm = mass[1]
    for c in range(52):
        cm[c] = 0.0
    o_x = base + N
    o_y = base + 2 * N
    o_e = base + 3 * N
    tot = 0.0
    k = 0
    while k < N:
        ke = prep[o_e + k]
        if ke == k + 1:
            h = prep[base + k]
            x = prep[o_x + k]
            y = prep[o_y + k]
            r = reach[h]
            vx = cm[x]
            vy = cm[y]
            winb[h] = tot - vx - vy
            tieb[h] = 0.0
            tot += r
            cm[x] = vx + r
            cm[y] = vy + r
        else:
            for q in range(k, ke):
                winb[prep[base + q]] = tot - cm[prep[o_x + q]] - cm[prep[o_y + q]]
            for q in range(k, ke):
                r = reach[prep[base + q]]
                tot += r
                cm[prep[o_x + q]] += r
                cm[prep[o_y + q]] += r
            for q in range(k, ke):
                h = prep[base + q]
                tieb[h] = tot - cm[prep[o_x + q]] - cm[prep[o_y + q]] + reach[h] - winb[h]
        k = ke
    wl = win_pay - loss_pay
    tl = tie_pay - loss_pay
    for h in range(N):
        legal = tot - cm[c1[h]] - cm[c2[h]] + reach[h]
        out[h] = wl * winb[h] + loss_pay * legal + tl * tieb[h]


@njit(cache=True, fastmath=True)
def _forward(T, b, c, p, mode, regret, strat, reach, sig, tot, N):
    ntype = T[0]
    actor = T[1]
    nact = T[2]
    children = T[3]
    reg_off = T[4]
    g0 = T[12][b]
    nb = T[13][b]
    for l in range(nb):
        g = g0 + l
        if ntype[g] != DECISION:
            continue
        A = nact[g]
        if actor[g] != p:
            if mode == 0:
                _sigma(regret, reg_off[g], c, A, N, sig, tot)
            else:
                _sigma(strat, reg_off[g], c, A, N, sig, tot)
            for a in range(A):
                ch = children[g, a]
                for h in range(N):
                    reach[ch, h] = reach[l, h] * sig[a, h]
        else:
            for a in range(A):
                ch = children[g, a]
                for h in range(N):
                    reach[ch, h] = reach[l, h]


@njit(cache=True, fastmath=True)
def _simple_leaves(T, b, c, p, reach, val, N, mass, prefix, held):
    ntype = T[0]
    pay = T[5]
    prep_off = T[11]
    prep_flat = T[15]
    c1 = T[16]
    c2 = T[17]
    g0 = T[12][b]
    nb = T[13][b]
    for l in range(nb):
        g = g0 + l
        t = ntype[g]
        if t == FOLD:
            _fold_leaf(reach[l], c1, c2, N, pay[g, p, 0], val[l], mass[0])
        elif t == SHOWDOWN:
            _showdown_leaf(prep_flat, prep_off[g] + c * 4 * N, reach[l], N, T,
                           pay[g, p, 0], pay[g, p, 1], pay[g, p, 2], val[l], prefix, held, mass)


@njit(cache=True, fastmath=True)
def _backward(T, b, c, p, mode, alpha, beta, gamma, regret, strat, val, sig, tot, N):
    ntype = T[0]
    actor = T[1]
    nact = T[2]
    children = T[3]
    reg_off = T[4]
    g0 = T[12][b]
    nb = T[13][b]
    for l in range(nb - 1, -1, -1):
        g = g0 + l
        if ntype[g] != DECISION:
            continue
        A = nact[g]
        v = val[l]
        if actor[g] != p:
            ch = children[g, 0]
            for h in range(N):
                v[h] = val[ch, h]
            for a in range(1, A):
                ch = children[g, a]
                for h in range(N):
                    v[h] += val[ch, h]
        elif mode == 1:
            ch = children[g, 0]
            for h in range(N):
                v[h] = val[ch, h]
            for a in range(1, A):
                ch = children[g, a]
                for h in range(N):
                    if val[ch, h] > v[h]:
                        v[h] = val[ch, h]
        elif mode == 2:
            _sigma(strat, reg_off[g], c, A, N, sig, tot)
            for h in range(N):
                v[h] = 0.0
            for a in range(A):
                ch = children[g, a]
                for h in range(N):
                    v[h] += sig[a, h] * val[ch, h]
        else:
            _sigma(regret, reg_off[g], c, A, N, sig, tot)
            for h in range(N):
                v[h] = 0.0
            for a in range(A):
                ch = children[g, a]
                for h in range(N):
                    v[h] += sig[a, h] * val[ch, h]
            base = reg_off[g] + c * A * N
            for a in range(A):
                ch = children[g, a]
                o = base + a * N
                for h in range(N):
                    r = regret[o + h]
                    regret[o + h] = r * (alpha if r >= 0.0 else beta) + (val[ch, h] - v[h])
                    strat[o + h] = strat[o + h] * gamma + sig[a, h]


@njit(cache=True)
def _make_pool(D, MAXN, MAXA, N, proto):
    return (np.empty((D, MAXN, N), proto.dtype), np.empty((D, MAXN, N), proto.dtype),
            np.empty((D, N), proto.dtype), np.empty((D, N), proto.dtype), np.empty((D, N), proto.dtype),
            np.empty((MAXA, N), proto.dtype), np.empty(N, proto.dtype),
            np.empty(N, np.float64), np.empty(N, np.float64), np.empty((2, 52), np.float64))


@njit(cache=True, fastmath=True)
def _child_reach(T, g, cc, reach_l, rc, lv, N):
    """Reach and live mask for dealt-card context ``cc`` of chance leaf ``g``."""
    draw = T[8][g]
    coff = T[10][g]
    cards = T[14]
    c1 = T[16]
    c2 = T[17]
    for h in range(N):
        ok = 1.0
        for t in range(draw):
            d = cards[coff + cc * draw + t]
            if c1[h] == d or c2[h] == d:
                ok = 0.0
        lv[h] = ok
        rc[h] = reach_l[h] * ok


@njit(cache=True, fastmath=True)
def _runout_leaves(T, b, c, p, reach, val, N, P, depth):
    """Chance leaves whose child is a bare showdown (an all-in before the river): inline."""
    reach_pool, val_pool, rc_pool, vc_pool, lv_pool, sig, tot, prefix, held, mass = P
    ntype = T[0]
    pay = T[5]
    ch_blk = T[6]
    ch_m = T[7]
    ch_w = T[9]
    prep_off = T[11]
    blk_off = T[12]
    prep_flat = T[15]
    g0 = blk_off[b]
    nb = T[13][b]
    rc = rc_pool[depth]
    vc = vc_pool[depth]
    lv = lv_pool[depth]
    for l in range(nb):
        g = g0 + l
        if ntype[g] != CHANCE:
            continue
        gs = blk_off[ch_blk[g]]
        if ntype[gs] != SHOWDOWN:
            continue
        m = ch_m[g]
        w = ch_w[g]
        for h in range(N):
            val[l, h] = 0.0
        for j in range(m):
            cc = c * m + j
            _child_reach(T, g, cc, reach[l], rc, lv, N)
            _showdown_leaf(prep_flat, prep_off[gs] + cc * 4 * N, rc, N, T,
                           pay[gs, p, 0], pay[gs, p, 1], pay[gs, p, 2], vc, prefix, held, mass)
            for h in range(N):
                val[l, h] += w * lv[h] * vc[h]


@njit(cache=True, fastmath=True)
def _eval_last(T, b, c, r_in, out, p, mode, alpha, beta, gamma, regret, strat, P, depth):
    """Serial evaluation of a block with no betting blocks below it (the river, or a runout)."""
    reach_pool, val_pool, rc_pool, vc_pool, lv_pool, sig, tot, prefix, held, mass = P
    N = r_in.shape[0]
    reach = reach_pool[depth]
    val = val_pool[depth]
    for h in range(N):
        reach[0, h] = r_in[h]
    _forward(T, b, c, p, mode, regret, strat, reach, sig, tot, N)
    _simple_leaves(T, b, c, p, reach, val, N, mass, prefix, held)
    _runout_leaves(T, b, c, p, reach, val, N, P, depth)
    _backward(T, b, c, p, mode, alpha, beta, gamma, regret, strat, val, sig, tot, N)
    for h in range(N):
        out[h] = val[0, h]


@njit(cache=True, fastmath=True)
def _eval_inner(T, b, c, r_in, out, p, mode, alpha, beta, gamma, regret, strat, P, depth):
    """Serial evaluation of a block that may continue into betting blocks one level below."""
    reach_pool, val_pool, rc_pool, vc_pool, lv_pool, sig, tot, prefix, held, mass = P
    N = r_in.shape[0]
    reach = reach_pool[depth]
    val = val_pool[depth]
    for h in range(N):
        reach[0, h] = r_in[h]
    _forward(T, b, c, p, mode, regret, strat, reach, sig, tot, N)
    _simple_leaves(T, b, c, p, reach, val, N, mass, prefix, held)
    _runout_leaves(T, b, c, p, reach, val, N, P, depth)
    ntype = T[0]
    ch_blk = T[6]
    ch_m = T[7]
    ch_w = T[9]
    blk_off = T[12]
    g0 = blk_off[b]
    nb = T[13][b]
    rc = rc_pool[depth]
    vc = vc_pool[depth]
    lv = lv_pool[depth]
    for l in range(nb):
        g = g0 + l
        if ntype[g] != CHANCE:
            continue
        cb = ch_blk[g]
        if ntype[blk_off[cb]] == SHOWDOWN:
            continue
        m = ch_m[g]
        w = ch_w[g]
        for h in range(N):
            val[l, h] = 0.0
        for j in range(m):
            cc = c * m + j
            _child_reach(T, g, cc, reach[l], rc, lv, N)
            _eval_last(T, cb, cc, rc, vc, p, mode, alpha, beta, gamma, regret, strat, P, depth + 1)
            for h in range(N):
                val[l, h] += w * lv[h] * vc[h]
    _backward(T, b, c, p, mode, alpha, beta, gamma, regret, strat, val, sig, tot, N)
    for h in range(N):
        out[h] = val[0, h]


@njit(cache=True, fastmath=True, parallel=True)
def eval_root(T, r_in, out, p, mode, alpha, beta, gamma, regret, strat,
              unit_l, unit_j, unit_out, nchunks, D, MAXN, MAXA):
    """Evaluate the root block (context 0); its chance children run in parallel.

    Pool levels: 0 root, 1 blocks below the root, 2 blocks below those.
    """
    N = r_in.shape[0]
    P0 = _make_pool(D, MAXN, MAXA, N, regret)
    reach_pool, val_pool, rc_pool, vc_pool, lv_pool, sig, tot, prefix, held, mass = P0
    reach = reach_pool[0]
    val = val_pool[0]
    for h in range(N):
        reach[0, h] = r_in[h]
    _forward(T, 0, 0, p, mode, regret, strat, reach, sig, tot, N)
    _simple_leaves(T, 0, 0, p, reach, val, N, mass, prefix, held)
    ntype = T[0]
    pay = T[5]
    ch_blk = T[6]
    ch_w = T[9]
    prep_off = T[11]
    blk_off = T[12]
    prep_flat = T[15]
    units = unit_l.shape[0]
    if units > 0:
        for ci in prange(nchunks):
            P = _make_pool(D, MAXN, MAXA, N, regret)
            rc = P[2][0]
            vc = P[3][0]
            lv = P[4][0]
            for u in range(ci, units, nchunks):
                g = unit_l[u]
                j = unit_j[u]
                cb = ch_blk[g]
                _child_reach(T, g, j, reach[g], rc, lv, N)
                gs = blk_off[cb]
                if ntype[gs] == SHOWDOWN:
                    _showdown_leaf(prep_flat, prep_off[gs] + j * 4 * N, rc, N, T,
                                   pay[gs, p, 0], pay[gs, p, 1], pay[gs, p, 2], vc, P[7], P[8], P[9])
                else:
                    _eval_inner(T, cb, j, rc, vc, p, mode, alpha, beta, gamma, regret, strat, P, 1)
                w = ch_w[g]
                for h in range(N):
                    unit_out[u, h] = w * lv[h] * vc[h]
        for l in range(T[13][0]):
            if ntype[l] == CHANCE:
                for h in range(N):
                    val[l, h] = 0.0
        for u in range(units):
            l = unit_l[u]
            for h in range(N):
                val[l, h] += unit_out[u, h]
    _backward(T, 0, 0, p, mode, alpha, beta, gamma, regret, strat, val, sig, tot, N)
    for h in range(N):
        out[h] = val[0, h]
