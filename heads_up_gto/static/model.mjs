export const ranks = 'AKQJT98765432';
export const suits = {s:'♠', h:'♥', d:'♦', c:'♣'};
export const deck = Object.keys(suits).flatMap(s => [...ranks].map(r => r+s));
export const samePath = (a,b) => a.length === b.length && a.every((v,i) => v === b[i]);

export function trainingLimit(mode, iterations, seconds) {
  if(mode === 'time') {
    if(!Number.isFinite(seconds) || seconds <= 0 || seconds > Number.MAX_SAFE_INTEGER)throw new Error('Enter a positive time limit in seconds.');
    return {time_limit_seconds:seconds};
  }
  if(mode !== 'iterations' || !Number.isSafeInteger(iterations) || iterations < 10)throw new Error('Enter a whole number of at least 10 iterations.');
  return {iterations};
}

// Inspecting a previous decision preserves its future; choosing a new action forks it.
export function branchPath(path, prefix, action) {
  if (prefix.some((v,i) => v !== path[i])) throw new Error('This decision is outside the current line.');
  const unchanged = path[prefix.length] === action;
  return {path: unchanged ? [...path] : [...prefix, action], focus: [...prefix, action], changed: !unchanged};
}

export function cellClass(i,j) {
  return i === j ? ranks[i]+ranks[j] : i < j ? ranks[i]+ranks[j]+'s' : ranks[j]+ranks[i]+'o';
}

export function handCombos(name) {
  const all = Object.keys(suits), out = [];
  for (let i=0;i<4;i++) for (let j=0;j<4;j++) {
    if (name[0] === name[1] ? j <= i : name[2] === 's' ? i !== j : i === j) continue;
    out.push(name[0]+all[i]+name[1]+all[j]);
  }
  return out;
}

export function comboKey(hand) { return [hand.slice(0,2),hand.slice(2,4)].sort().join(''); }
export function isBlocked(hand,board) { return board.includes(hand.slice(0,2)) || board.includes(hand.slice(2,4)); }

export function groupRows(rows, actions) {
  const groups = new Map();
  for (const row of rows) {
    if (!groups.has(row.class)) groups.set(row.class,{rows:[],mass:0,weight:0,covered:0,totals:Array(actions.length).fill(0)});
    const g = groups.get(row.class);
    g.rows.push(row);g.mass+=row.mass;g.weight+=row.weight;
    if (row.probabilities) {
      g.covered+=row.mass;
      row.probabilities.forEach((p,i)=>g.totals[i]+=p*row.mass);
    }
  }
  return groups;
}

export function chooseCard(selected, card, count, blocked) {
  if (!deck.includes(card) || blocked.includes(card)) return [...selected];
  if (selected.includes(card)) return selected.filter(c=>c!==card);
  return selected.length < count ? [...selected,card] : [...selected.slice(0,-1),card];
}

export function actionColors(actions) {
  const aggressive = actions.filter(a=>['bet','raise'].includes(a.kind));
  return actions.map(a => a.kind === 'fold' ? '#4387bd' : ['call','check'].includes(a.kind) ? '#58b568' :
    /all.in/i.test(a.label) ? '#7b252c' : ['#ed484b','#c5323a','#952a32'][Math.min(2,aggressive.indexOf(a))]);
}

// Keep original indices so display order never changes the strategy's action mapping.
export function actionDisplay(actions) {
  const colors = actionColors(actions);
  const priority = a => /all.in|jam/i.test(a.label || '') ? 3 : ['bet','raise'].includes(a.kind) ? 2 : a.kind === 'fold' ? 0 : 1;
  return actions.map((a,index)=>({...a,index,color:colors[index]}))
    .sort((a,b)=>priority(b)-priority(a) || (b.target ?? b.amount ?? 0)-(a.target ?? a.amount ?? 0));
}

export function actionGradient(probabilities, display) {
  return mixGradient(display.map(a=>probabilities[a.index] ?? 0),display.map(a=>a.color));
}

export function mixGradient(probabilities, colors, remainder='#404347') {
  let start=0;
  const stops=[];
  probabilities.forEach((p,i)=>{const end=start+100*p;if(end>start)stops.push(`${colors[i]} ${start}% ${end}%`);start=end;});
  if (start<99.999) stops.push(`${remainder} ${start}% 100%`);
  return stops.length ? `linear-gradient(to right,${stops.join(',')})` : remainder;
}
