"""Public-state navigation for the horizontal preflop action strip."""
from .cards import COMBOS, combo_text, hand_class
from .preflop import canonical_action, validate_chart


def action_label(key):
    if key.startswith('raise:'):
        amount = float(key.split(':')[1])
        return f'All-in {amount:g}' if amount == 100 else f'Raise {amount:g}'
    return key.capitalize()


def preflop_view(history, chart=None):
    if not isinstance(history, list) or len(history) > 30 or any(not isinstance(a, str) for a in history):
        raise ValueError('Preflop history must be a list of at most 30 actions.')
    history = [canonical_action(a) for a in history]
    compiled = validate_chart(chart) if chart is not None else None
    ranges = [dict.fromkeys(COMBOS, 1.0), dict.fromkeys(COMBOS, 1.0)]
    paid, actor, last_raise = [1., .5], 1, 1.
    timeline, prefix = [], []
    closed, terminal, winner = False, '', -1

    def available():
        if compiled is not None:
            return list(compiled.get('/'.join(prefix), {}))
        facing = paid[1 - actor] - paid[actor]
        keys = ['fold', 'call'] if facing > 0 else ['check']
        target = 2.5 if not prefix else 11.5 if max(paid) <= 2.5 else 26 if max(paid) <= 11.5 else 100
        if target > max(paid):
            keys.append(f'raise:{target:g}')
        if max(paid) < 100 and 'raise:100' not in keys:
            keys.append('raise:100')
        return keys

    def actions(keys):
        return [{'key': a, 'kind': a.split(':')[0], 'target': float(a.split(':')[1]) if ':' in a else 0, 'amount': 0, 'label': action_label(a)} for a in keys]

    def panel(selected=None):
        keys = available()
        if compiled is None and selected and selected not in keys:
            keys.append(selected)  # explicit manual sizes still undergo legality checks
        return {'kind': 'decision', 'phase': 'preflop', 'street': 'Preflop', 'actor': actor,
                'stack': round(100 - paid[actor], 2), 'pot': sum(paid), 'actions': actions(keys),
                'path': list(prefix), 'selected': selected}

    for index, action in enumerate(history):
        if closed or terminal:
            raise ValueError('The preflop line continues after the hand or round has ended.')
        if compiled is not None and action not in available():
            raise ValueError('No chart is supplied for that preflop branch. Return to an earlier action or use manual ranges.')
        timeline.append(panel(action))
        if compiled is not None:
            policy = compiled['/'.join(prefix)][action]
            ranges[actor] = {h: w * policy.get(h, 0) for h, w in ranges[actor].items() if w * policy.get(h, 0) > 0}
        other = 1 - actor
        facing = paid[other] - paid[actor]
        if action == 'fold':
            if facing <= 0:
                raise ValueError('There is no bet to fold to.')
            terminal, winner = 'fold', other
        elif action == 'call':
            if facing <= 0:
                raise ValueError('There is no bet to call.')
            paid[actor] = paid[other]
            closed = index > 0
        elif action == 'check':
            if facing > 0 or not (index == 1 and prefix == ['call']):
                raise ValueError('A preflop check is only legal for BB after an SB limp.')
            closed = True
        else:
            target = float(action.split(':')[1])
            increase = target - max(paid)
            if increase <= 0 or (increase < last_raise - 1e-8 and target != 100):
                raise ValueError('That preflop raise is below the minimum raise size.')
            paid[actor] = target
            last_raise = max(last_raise, increase)
        prefix.append(action)
        actor = other
    if closed and max(paid) == 100:
        terminal = 'showdown'
    missing = not closed and not terminal and not available()
    if not closed and not terminal:
        timeline.append(panel())
    current = actions(available()) if not closed and not terminal else []
    response = {'street': 'Preflop', 'phase': 'preflop', 'board': [], 'actor': actor,
                'actor_label': 'BB · out of position' if actor == 0 else 'SB · button',
                'pot': sum(paid), 'stacks': [round(100 - p, 2) for p in paid], 'path': history,
                'pending': closed and not terminal, 'terminal': terminal, 'winner': winner,
                'actions': current, 'timeline': timeline, 'rows': [], 'aggregate': [],
                'coverage': 0., 'fallback_used': False, 'missing_chart': missing,
                'manual': compiled is None, 'source_name': chart.get('name', 'Imported chart') if chart else 'Explicit ranges'}
    if not current or compiled is None:
        return response
    policies = compiled['/'.join(prefix)]
    opposite = ranges[1 - actor]
    other_total = sum(opposite.values())
    card_mass = [0.] * 52
    for (a, b), w in opposite.items():
        card_mass[a] += w
        card_mass[b] += w
    total, covered, aggregate = 0., 0., [0.] * len(current)
    for h, w in ranges[actor].items():
        mass = w * max(0., other_total - card_mass[h[0]] - card_mass[h[1]] + opposite.get(h, 0))
        if mass <= 1e-12:
            continue
        probs = [policies[a['key']].get(h, 0) for a in current]
        # Imported charts may omit residual frequencies. Do not invent a mix.
        if abs(sum(probs) - 1.) > 1e-6:
            probs = None
        total += mass
        response['rows'].append({'hand': combo_text(h), 'class': hand_class(h), 'weight': w, 'mass': mass,
                                 'probabilities': probs, 'updates': None, 'averages': None})
        if probs is not None:
            covered += mass
            for i, p in enumerate(probs):
                aggregate[i] += mass * p
    if total:
        for row in response['rows']:
            row['mass'] /= total
        response['aggregate'] = [v / covered for v in aggregate] if covered else []
        response['coverage'] = covered / total
    return response
