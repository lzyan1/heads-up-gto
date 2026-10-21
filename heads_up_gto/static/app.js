import {ranks,suits,deck,samePath,branchPath,cellClass,handCombos,comboKey,isBlocked,groupRows,chooseCard,actionDisplay,actionGradient,trainingLimit} from './model.mjs';
const $ = id => document.getElementById(id);
const fmt = n => Number(n).toLocaleString(undefined,{maximumFractionDigits:2});
const pct = n => `${(100*n).toFixed(1)}%`;
const suitColors = {s:'#555b61',h:'#bb282d',d:'#334fd3',c:'#258c48'};
const suitNames = {s:'spades',h:'hearts',d:'diamonds',c:'clubs'};
const player = actor => actor===0?'BB':'SB';
const deepFields = [['deep-device','deep_device'],['deep-hidden','deep_hidden'],['deep-buffer-size','deep_buffer_size'],['deep-traversals','deep_traversals'],['deep-train-steps','deep_train_steps'],['deep-batch-size','deep_batch_size'],['deep-inference-batch','deep_inference_batch']];
let config=null, prePath=[], preEnd=null, node=null, phase='preflop';
let ctx=null, contexts=new Map(), loading=false, pollTimer=null, view='strategy', pinned=null, hovered=null, groups=new Map(), picker=null;
let comparison=null;
const locked = () => loading || !!ctx?.job.busy;
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function button(text,cls,handler){const b=el('button',cls,text);b.type='button';b.onclick=handler;return b;}
function navButton(text,cls,handler){const b=button(text,cls,handler);b.dataset.navigation='';b.disabled=locked();return b;}
function error(e,target='error'){$(target).textContent=e.message||String(e);$(target).hidden=false;}
function clearError(){$('error').hidden=true;}
async function api(url,body){const response=await fetch(url,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed.');return data;}
async function task(fn,target='error'){
  if(loading)return;loading=true;clearError();controls();
  try{await fn();}catch(e){error(e,target);}finally{loading=false;controls();}
}
function card(c,mini=false){
  if(mini){const e=el('span','mini-card',c[0]+suits[c[1]]);e.style.setProperty('--suit-color',suitColors[c[1]]);return e;}
  const e=el('span',`playing-card${c?'':' empty'}`);e.setAttribute('aria-label',c?`${c[0]} of ${suitNames[c[1]]}`:'Empty card');
  if(c){e.style.setProperty('--suit-color',suitColors[c[1]]);e.dataset.suit=suits[c[1]];e.append(el('span','rank',c[0]),el('span','suit-symbol',suits[c[1]]));}else e.textContent='?';return e;
}
function cards(cards){const e=el('div','board');cards.forEach(c=>e.append(card(c)));return e;}
function shortAction(a){return a.label.replace(/ BB.*/,'').replace('Raise to','Raise');}
function readConfig(){
  return {...trainingLimit($('stop-mode').value,Number($('iterations').value),Number($('time-limit').value)),settings:{
    backend:$('backend').value,
    ...Object.fromEntries(deepFields.map(([id,key])=>[key,id==='deep-device'?$(id).value:Number($(id).value)])),
    mode:$('mode').value,chance_mode:$('chance-mode').value,buckets:Number($('buckets').value),equity_samples:Number($('samples').value),seed:Number($('seed').value),
    max_nodes:$('max-nodes').value.trim()===''?null:Number($('max-nodes').value)
  }};
}
function updateSettingsFields(){
  const deep=$('backend').value==='deep_cfr';
  if(deep){$('mode').value='exact';$('chance-mode').value='sampled';$('max-nodes').value='';}
  $('deep-help').hidden=!deep;$('deep-settings').hidden=!deep;
  for(const [id] of deepFields)$(id).disabled=!deep;
  $('chance-mode').disabled=deep;$('max-nodes').disabled=deep;
  document.querySelector('label[for="iterations"]').textContent=deep?'Neural rounds per run':'Iterations per run';
  const enumerated=$('chance-mode').value==='enumerate_turns';
  if(enumerated)$('mode').value='exact';
  $('mode').disabled=enumerated||deep;
  const timed=$('stop-mode').value==='time',exact=$('mode').value==='exact';
  $('iteration-field').hidden=timed;$('iterations').disabled=timed;$('time-field').hidden=!timed;$('time-limit').disabled=!timed;
  $('bucket-settings').hidden=exact;$('buckets').disabled=exact;$('samples').disabled=exact;
  $('card-model-help').textContent=deep?'Learns from exact cards and betting history to predict strategies for related states.':exact?'Tracks each card combination separately. Exact cards still require training to converge.':'Hands in the same strength bucket share a strategy.';
}
function restoreSettings(){
  const timed=config.time_limit_seconds!==undefined;
  $('stop-mode').value=timed?'time':'iterations';
  $(timed?'time-limit':'iterations').value=timed?config.time_limit_seconds:config.iterations;
  for(const [id,key] of [['chance-mode','chance_mode'],['mode','mode'],['buckets','buckets'],['samples','equity_samples'],['seed','seed'],['max-nodes','max_nodes']])$(id).value=config.settings[key]??'';
  $('backend').value=config.settings.backend||'tabular';
  for(const [id,key] of deepFields)if(config.settings[key]!==undefined)$(id).value=config.settings[key];
  updateSettingsFields();
}

const preRequest = history => api('/api/preflop',{history});
function saveContext(job,board,baseTrail=[],rootBoard=board){
  ctx={job,board:[...board],rootBoard:[...rootBoard],path:[],end:null,baseTrail};contexts.set(job.id,ctx);
  while(contexts.size>2)contexts.delete(contexts.keys().next().value);
}
async function resetHand(cfg=config){
  const root=await preRequest([]);config=cfg;prePath=[];preEnd=root;node=root;phase='preflop';ctx=null;contexts.clear();pinned=null;hovered=null;clearTimeout(pollTimer);render(true);
}
async function inspectPre(path){const result=await preRequest(path);phase='preflop';node=result;render();}
async function choosePre(prefix,action){
  const next=branchPath(prePath,prefix,action),result=await preRequest(next.focus);
  if(next.changed){prePath=next.path;preEnd=result;ctx=null;contexts.clear();}
  if(result.pending&&!next.changed&&ctx){await inspectPost(ctx,[]);return;}
  phase='preflop';node=result;render(true);
  if(result.pending)openPicker({street:'Flop',count:3,blocked:[],selected:[],prefix:[]});
}
async function inspectPost(target,path){
  const result=await api(`/api/jobs/${target.job.id}/node`,{path});ctx=target;phase='postflop';node=result;
  if(!ctx.end)ctx.end=result;
  render();
}
async function choosePost(target,prefix,action){
  const next=branchPath(target.path,prefix,action);
  const result=await api(`/api/jobs/${target.job.id}/node`,{path:next.focus});
  ctx=target;ctx.path=next.path;if(next.changed||!ctx.end)ctx.end=result;
  node=result;phase='postflop';render(true);
  if(result.pending){const previous=ctx.path[result.path.length];openPicker({street:result.street==='Flop'?'Turn':'River',count:1,blocked:result.board,selected:previous?.startsWith('deal:')?[previous.slice(5)]:[],prefix:result.path,context:ctx});}
}
async function startSolve(board,cfg=config,history=prePath,pre=preEnd){
  const job=await api('/api/solve',{...cfg,preflop:history.join('/'),flop:board.join(' ')});
  config=cfg;prePath=[...history];preEnd=pre;saveContext(job,board);phase='postflop';node=null;pinned=null;hovered=null;
  render(true);poll(job.id);
}
async function poll(id){
  clearTimeout(pollTimer);
  try{
    const target=contexts.get(id);if(!target)return;
    target.job=await api(`/api/jobs/${id}`);if(ctx===target)controls();
    if(target.job.busy){pollTimer=setTimeout(()=>poll(id),650);return;}
    if(target.job.stats.error)throw new Error(target.job.stats.error);
    if(ctx===target){if(loading){pollTimer=setTimeout(()=>poll(id),150);return;}await task(()=>inspectPost(target,node?.phase==='postflop'?node.path:[]));}
  }catch(e){error(e);controls();}
}
function controls(){
  const busy=!!ctx?.job.busy;
  if(picker){
    document.querySelectorAll('#deck button').forEach(b=>b.disabled=loading||picker.blocked.includes(b.dataset.card));
    $('confirm-cards').disabled=loading||picker.selected.length!==picker.count;
    for(const id of ['card-close','clear-cards','random-cards'])$(id).disabled=loading;
  }
  document.querySelectorAll('[data-navigation]').forEach(b=>b.disabled=locked()||b.dataset.expired==='true');
  $('new-hand').disabled=locked();
  $('settings-open').disabled=false;
  $('load-spot').disabled=locked();
  $('reference-preset').disabled=locked();
  $('stop').hidden=!busy;$('stop').disabled=loading;
  const hasSolve=!!ctx&&phase==='postflop';
  $('resume').hidden=!hasSolve||busy;$('resume').disabled=loading;
  $('refine').hidden=!hasSolve||busy||!node||!!node.pending||!!node.terminal;$('refine').disabled=loading;
  $('export').hidden=!hasSolve||busy||!node;$('export').disabled=loading;
  $('compare').hidden=!hasSolve||busy||!ctx?.job.metadata.reference_line_available;$('compare').disabled=loading;
  if(busy){const s=ctx.job.stats,timed=s.time_limit_seconds!=null;
    const unit=s.backend==='deep_cfr'?'neural rounds':'iterations';
    const phaseLabel={sampling_0:'Sampling BB',sampling_1:'Sampling SB',advantage_0:'Fitting BB advantages',advantage_1:'Fitting SB advantages',average_0:'Fitting BB average',average_1:'Fitting SB average'}[s.phase];
    $('study-status').textContent=(timed?`Solving · ${fmt(s.run_seconds||0)} / ${fmt(s.time_limit_seconds)}s · ${fmt(s.iterations||0)} ${unit}`:`Solving · ${fmt(s.iterations||0)} / ${fmt(s.target||0)} ${unit}`)+(phaseLabel?` · ${phaseLabel} ${s.phase_step}/${s.phase_total}`:'');
    $('progress').style.width=`${Math.min(100,100*(timed?(s.run_seconds||0)/s.time_limit_seconds:(s.iterations||0)/(s.target||1)))}%`;
  }else{$('progress').style.width='0';const s=ctx?.job.stats;
    const suffix=s?.status==='cancelled'?' · stopped':s?.status==='memory_exhausted'?' · RAM full · partial results saved':s?.status==='node_limit'?' · decision limit reached':s?.status==='complete'&&s.time_limit_seconds!=null?' · time limit reached':'';
    $('study-status').textContent=loading?'Loading decision…':hasSolve?`${fmt(s.iterations||0)} ${s.backend==='deep_cfr'?'neural rounds':'iterations'} · ${fmt(s.seconds||0)}s${suffix}${s.pending_round?' · round paused; Train more to continue':''}`:'GTO Wizard · hardcoded preflop';
  }
}
function addDecision(panel,jobId){
  const pre=panel.phase==='preflop',target=pre?null:contexts.get(jobId);
  const active=node&&phase===(pre?'preflop':'postflop')&&(pre||target===ctx)&&samePath(node.path,panel.path);
  const tile=el('div',`decision-tile${active?' active':''}`);
  const heading=navButton('', 'tile-heading',()=>task(()=>pre?inspectPre(panel.path):inspectPost(target,panel.path)));
  heading.append(el('span','',player(panel.actor)),el('span','',fmt(panel.stack)));
  heading.setAttribute('aria-label',`View ${panel.street} ${player(panel.actor)} decision ${panel.path.length+1}`);
  if(!pre&&!target){heading.dataset.expired='true';heading.disabled=true;heading.title='An older solve is no longer in memory.';}
  tile.append(heading);
  for(const a of panel.actions){const b=navButton(shortAction(a),`tile-action${panel.selected===a.key?' chosen':''}`,()=>task(()=>pre?choosePre(panel.path,a.key):choosePost(target,panel.path,a.key)));
    b.setAttribute('aria-label',`${player(panel.actor)} ${shortAction(a)}, ${panel.street}, decision ${panel.path.length+1}`);b.setAttribute('aria-pressed',String(panel.selected===a.key));
    if(!pre&&!target){b.dataset.expired='true';b.disabled=true;}tile.append(b);}
  if(!panel.actions.length)tile.append(el('small','','No chart for this branch'));
  $('action-strip').append(tile);
}
function addBoard(street,board,pot,handler,pending=false){
  const b=navButton('',`board-tile${pending?' pending':''}`,handler);
  const heading=el('div','board-heading');heading.append(el('span','',street),el('span','',`${fmt(pot)} BB`));
  b.append(heading,cards(board),el('small','',pending?'Choose cards ↗':'Edit board ↗'));
  b.setAttribute('aria-label',`${pending?'Choose':'Edit'} ${street.toLowerCase()} cards`);$('action-strip').append(b);
}
function renderStrip(scroll=false){
  const strip=$('action-strip'),position=strip.scrollLeft;strip.replaceChildren();
  const info=el('div','game-tile');info.append(el('strong','','Cash · 100 BB'),el('span','','Heads-up · no rake'),el('small','','0.5 / 1 blinds · SB is button'));strip.append(info);
  for(const p of preEnd?.timeline||[])addDecision(p);
  if(ctx){
    addBoard('Flop',ctx.board,preEnd.pot,()=>openPicker({street:'Flop',count:3,blocked:[],selected:ctx.board,prefix:[]}));
    const panels=[...ctx.baseTrail,...(ctx.end?.timeline||[]).map(p=>({...p,jobId:ctx.job.id}))];
    for(const p of panels){
      if(p.kind==='decision'){addDecision(p,p.jobId);continue;}
      const target=contexts.get(p.jobId);
      addBoard(p.street,p.board.slice(-1),p.pot,()=>{
        if(!target){error(new Error('This older solve is no longer in memory. Edit the flop to start a new line.'));return;}
        openPicker({street:p.street,count:1,blocked:p.board.slice(0,-1),selected:p.board.slice(-1),prefix:p.path,context:target});
      });
    }
    if(ctx.end?.pending){const n=ctx.end;addBoard(n.street==='Flop'?'Turn':'River',[null],n.pot,()=>openPicker({street:n.street==='Flop'?'Turn':'River',count:1,blocked:n.board,selected:[],prefix:n.path,context:ctx}),true);}
  }else if(preEnd?.pending)addBoard('Flop',[null,null,null],preEnd.pot,()=>openPicker({street:'Flop',count:3,blocked:[],selected:[],prefix:[]}),true);
  const end=ctx?.end||preEnd;
  if(end?.terminal){const tile=el('div','tile-end');tile.append(el('b','',end.terminal==='fold'?`${player(end.winner)} wins`:'Showdown / all-in'),el('span','',`${fmt(end.pot)} BB pot`));strip.append(tile);}
  if(scroll)requestAnimationFrame(()=>{const active=strip.querySelector('.active');if(active){const a=active.getBoundingClientRect(),s=strip.getBoundingClientRect();if(a.right>s.right)strip.scrollLeft+=a.right-s.right;if(a.left<s.left)strip.scrollLeft-=s.left-a.left;}else strip.scrollLeft=strip.scrollWidth;});else strip.scrollLeft=position;
}
function renderGrid(){
  groups=groupRows(node?.rows||[],node?.actions||[]);const display=actionDisplay(node?.actions||[]);$('range-grid').replaceChildren();
  for(let i=0;i<13;i++)for(let j=0;j<13;j++){
    const name=cellClass(i,j),g=groups.get(name),count=handCombos(name).length,reach=Math.min(1,(g?.weight||0)/count);
    const b=button('',`range-cell${g?'':' absent'}${pinned===name?' pinned':''}`,()=>{pinned=pinned===name?null:name;renderSelection(name);});
    b.dataset.hand=name;b.setAttribute('aria-label',`${name}${g?`, ${g.rows.length} available combinations`:': outside current range'}`);b.setAttribute('aria-pressed',String(pinned===name));
    const mix=el('span','cell-mix');mix.style.height=`${view==='strategy'?100:100*reach}%`;
    if(g){mix.style.background=view==='ranges'?'#58b568':actionGradient(g.totals.map(v=>g.mass?v/g.mass:0),display);}
    else mix.style.background='#202224';
    b.append(mix,el('span','cell-label',name));
    b.onmouseenter=()=>{hovered=name;renderHand(name);};b.onfocus=()=>{hovered=name;renderHand(name);};
    b.onkeydown=e=>{const deltas={ArrowRight:1,ArrowLeft:-1,ArrowDown:13,ArrowUp:-13};if(deltas[e.key]){e.preventDefault();const n=i*13+j+deltas[e.key];$('range-grid').children[Math.max(0,Math.min(168,n))].focus();}if(e.key==='Escape'){pinned=null;renderSelection();}};
    $('range-grid').append(b);
  }
  $('legend').replaceChildren();
  const entries=view==='ranges'?[{name:'Range reach',color:'#58b568'}]:display.map(a=>({name:shortAction(a),color:a.color}));
  for(const {name,color} of entries){const s=el('span'),dot=el('i');dot.style.background=color;s.append(dot,document.createTextNode(name));$('legend').append(s);}
  const absent=el('span'),dot=el('i');dot.style.background='#404347';absent.append(dot,document.createTextNode('Untrained / absent'));$('legend').append(absent);
}
function renderSelection(name){
  document.querySelectorAll('.range-cell').forEach(b=>{b.classList.toggle('pinned',b.dataset.hand===pinned);b.setAttribute('aria-pressed',String(b.dataset.hand===pinned));});renderHand(name||pinned||hovered);
}
function renderHand(name){
  const grid=$('combo-grid');grid.replaceChildren();$('unpin').hidden=!pinned;$('hand-summary').textContent='';
  $('hand-title').textContent=name||'Hands';$('hand-subtitle').textContent=name?(pinned===name?'Pinned hand':'Individual combinations'):'Hover over the range';
  if(!name){const empty=el('div','empty-detail');empty.append(el('span','','↖'),el('strong','','A closer look at every hand.'),el('p','','Hover over the range to see each suit combination and its action frequencies.'));grid.append(empty);return;}
  const g=groups.get(name),map=new Map((g?.rows||[]).map(r=>[comboKey(r.hand),r])),display=actionDisplay(node?.actions||[]);
  if(g){$('hand-summary').textContent=`${fmt(g.weight)} weighted combos · ${pct(g.mass)} of this range${g.covered<g.mass*.999?' · some combinations are untrained':''}`;}
  else $('hand-summary').textContent=node?.pending?'Choose the next board card to see its strategy.':node?.terminal?'No decisions remain in this hand.':ctx?.job.busy?'Training this continuation…':'This hand is outside the current range or blocked by the board.';
  for(const hand of handCombos(name)){
    const blocked=isBlocked(hand,node?.board||[]),r=map.get(comboKey(hand)),box=el('div',`combo-card${blocked?' blocked':''}`);
    const heading=el('div','combo-heading'),hole=el('div','hole-cards');hole.append(card(hand.slice(0,2),true),card(hand.slice(2,4),true));heading.append(hole,el('span','combo-weight',r?`${Math.round(100*r.weight)}% reach`:''));box.append(heading);
    if(blocked||!r||!r.probabilities){box.append(el('div','combo-unavailable',blocked?'Board blocked':!r?'Outside range':phase==='preflop'?'Chart incomplete':'Not trained yet'));}
    else{
      display.forEach(a=>{const row=el('div','combo-frequency');row.append(el('span','',shortAction(a)),el('b','',pct(r.probabilities[a.index])));box.append(row);});
      const bar=el('div','combo-bar');bar.style.background=actionGradient(r.probabilities,display);box.append(bar);
      if(r.updates!==null){box.title=`${node?.neural?'Neural prediction · ':''}${fmt(r.updates)} ${node?.neural?'advantage samples':'regret updates'} · ${fmt(r.averages)} average samples${r.updates<200||r.averages<200?' · limited direct sampling':''}`;}
    }
    if(phase==='postflop'&&r)box.append(el('div','combo-unavailable',`${node?.neural?'Prediction · ':''}${fmt(r.updates)} ${node?.neural?'advantage samples':'regret updates'} · ${fmt(r.averages)} average samples`));
    grid.append(box);
  }
}
function renderActions(){
  $('action-list').replaceChildren();$('decision-message').hidden=true;$('action-bar').hidden=false;
  const current=node?.actions||[],display=actionDisplay(current);
  $('action-list').style.gridTemplateColumns=`repeat(${Math.max(1,display.length)},minmax(0,1fr))`;
  $('action-list').classList.toggle('compact',display.length>2);
  for(const a of display){
    const b=navButton('','action-card',()=>task(()=>phase==='preflop'?choosePre(node.path,a.key):choosePost(ctx,node.path,a.key)));
    b.style.setProperty('--action-color',a.color);b.setAttribute('aria-label',`${shortAction(a)} at current decision`);
    const known=!!node.aggregate.length;
    b.append(el('span','action-name',shortAction(a)),el('span',`action-freq${known?'':' unknown'}`,known?pct(node.aggregate[a.index]):phase==='preflop'?'No chart':'Untrained'));
    if(a.kind==='bet'&&!/all.in/i.test(a.label))b.append(el('span','action-extra',`${Math.round(100*a.amount/node.pot)}% pot`));
    $('action-list').append(b);
  }
  $('action-bar').style.background=actionGradient(node?.aggregate||[],display);
  $('action-hint').textContent=node?.neural?'Predicted average strategy':node?.coverage>0&&node.coverage<.999?'Averages of trained hands':'Click to continue';
  if(!current.length){
    const box=$('decision-message');box.hidden=false;box.replaceChildren();$('action-bar').hidden=true;
    if(ctx?.job.busy){box.append(el('strong','','Solving the continuation…'),el('span','','Your board and ranges are being trained across the remaining streets. You can stop early to explore the partial result.'));}
    else if(node?.pending){const street=phase==='preflop'?'Flop':node.street==='Flop'?'Turn':'River';box.append(el('strong','',`Ready for the ${street.toLowerCase()}.`),el('span','','The betting round is complete. Choose the next board cards to continue.'),navButton(`Choose ${street.toLowerCase()}`,'primary',()=>openPicker({street,count:street==='Flop'?3:1,blocked:node.board,selected:[],prefix:node.path,context:phase==='postflop'?ctx:null})));}
    else if(node?.terminal){box.append(el('strong','',node.terminal==='fold'?`${player(node.winner)} wins the pot.`:'No more betting decisions.'),el('span','','Return to an earlier decision in the action strip to explore another line.'));}
    else if(node?.missing_chart){box.append(el('strong','','This branch needs another chart.'),el('span','','The supplied screenshots do not contain this response. Choose an earlier action from the hardcoded chart.'));}
    else box.append(el('strong','','Choose a decision above.'));
  }
}
function render(scroll=false){
  renderStrip(scroll);controls();
  const board=node?.board||(phase==='postflop'?ctx?.rootBoard:[])||[],n=node;
  $('board').replaceChildren(...board.map(c=>card(c)));$('pot').textContent=`${fmt(n?.pot??preEnd?.pot??1.5)} BB`;
  $('street-label').textContent=n?.street||(phase==='postflop'?({3:'Flop',4:'Turn',5:'River'}[board.length]):'Preflop');$('range-street').textContent=$('street-label').textContent;
  const stacks=n?.stacks||ctx?.job.metadata.stacks||[99,99.5];$('bb-stack').textContent=fmt(stacks[0]);$('sb-stack').textContent=fmt(stacks[1]);
  $('bb-badge').classList.toggle('to-act',!!n?.actions.length&&n.actor===0);$('sb-badge').classList.toggle('to-act',!!n?.actions.length&&n.actor===1);
  $('range-player').textContent=n?.actions.length?`${player(n.actor)} to act`:'Range';
  $('range-caption').textContent=view==='breakdown'?'Bar height = reach · color = action':view==='ranges'?'Bar height = range reach':'Conditional action frequencies';
  $('coverage').textContent=n?.phase==='postflop'&&n.actions.length?`${pct(n.coverage)} ${n.neural?'direct samples':'with samples'}`:n?.actions.length&&!n.manual?(n.coverage<.999?`${pct(n.coverage)} chart coverage`:'Chart strategy'):'';
  $('source-label').textContent='GTO Wizard · hardcoded preflop';
  let note=phase==='preflop'?'Preflop frequencies are rounded to 10%. SB’s 4-bet jam and BB’s 5-bet jam are allowed. Opening and 3-bet jams are disabled.':'One scenario bet or raise size, plus all-in whenever legal. Heuristic sizes; approximate strategy, not a GTO certificate.';
  if(phase==='postflop'&&ctx?.job.metadata.settings.chance_mode==='enumerate_turns')note+=' Experimental turn enumeration: all legal turns per sampled hand pair, one sampled river each; private hands and opponent actions still sampled.';
  if(phase==='postflop'&&n?.sizing)note+=` ${n.sizing.reason}`;
  if(phase==='preflop'&&n?.actions.length&&!n.manual&&n.coverage<.999)note+=' This chart omits some frequencies; averages include only hands with a complete mix.';
  if(phase==='postflop'&&ctx?.job.metadata.settings.mode==='bucketed')note+=' Hands in the same strength bucket share frequencies.';
  if(n?.coverage<.999&&phase==='postflop'&&n.actions.length&&!n.neural)note+=' Action totals include only hands with both regret updates and average samples. Refine this decision to train this board directly.';
  if(n?.neural){
    note+=` Deep CFR: ${pct(n.prediction_coverage)} of range mass has neural predictions. Action totals include predicted unseen hands. Direct sample coverage is separate and does not measure prediction accuracy.`;
    if(!ctx?.job.stats.prediction_ready)note+=' No complete neural round yet; choose Train more to finish the first average strategy.';
    if(n.neural.tracking_saturated)note+=' The visit-tracking limit was reached; direct sample counts are lower bounds.';
  }
  if(phase==='postflop'&&(n?.fallback_used||ctx?.job.metadata.fallback_used))note+=' An untrained earlier decision used a uniform fallback to carry ranges forward.';
  if(ctx?.job.metadata.local_refinement&&phase==='postflop')note+=' This is a local re-solve, without a full-game equilibrium guarantee.';
  if(phase==='postflop'&&n?.training&&!n.neural){
    const t=n.training,c=t.counts;
    note+=` ${c.unseen} unseen hands; ${c.regret_only} regret-only; ${c.average_only} average-only; ${c.low_samples} with fewer than ${t.sample_threshold} of either sample. Median regret updates / average samples: ${fmt(t.regret_updates.median)} / ${fmt(t.average_samples.median)}. Counts do not certify convergence.`;
  }
  if(phase==='postflop'&&ctx?.job.stats.partial_updates_possible)note+=` ${ctx.job.stats.interrupted_iterations} interrupted iteration(s) may have retained partial updates; the iteration count includes only completed pairs of traversals.`;
  $('result-note').textContent=note;
  renderGrid();renderActions();renderHand(pinned||hovered);controls();
}
function openPicker(options){
  if(ctx?.job.busy)return;
  picker={...options,selected:[...options.selected]};$('card-error').hidden=true;
  $('card-title').textContent=`Choose the ${picker.street.toLowerCase()}`;$('card-eyebrow').textContent=picker.selected.length?'BOARD SELECTION':'NEXT STREET';
  $('card-help').textContent=picker.count===3?'Choose three cards. Confirming starts a solve for this flop.':'Choose one card. Dealt board cards are unavailable.';
  $('confirm-cards').textContent=picker.count===3?'Solve continuation':`View ${picker.street.toLowerCase()}`;
  renderPicker();$('card-dialog').showModal();
}
function renderPicker(){
  $('selected-cards').replaceChildren();for(let i=0;i<picker.count;i++){
    const c=picker.selected[i],b=button('','',()=>{if(c){picker.selected=picker.selected.filter(x=>x!==c);renderPicker();}});b.append(card(c));b.setAttribute('aria-label',c?`Remove ${c}`:'Empty selection');b.disabled=!c;$('selected-cards').append(b);
  }
  $('deck').replaceChildren();for(const c of deck){const b=button('','',()=>{picker.selected=chooseCard(picker.selected,c,picker.count,picker.blocked);renderPicker();const selected=$('deck').querySelector(`[data-card="${c}"]`);selected?.focus();});b.dataset.card=c;b.append(card(c));b.disabled=picker.blocked.includes(c);b.setAttribute('aria-label',`${c[0]} of ${suitNames[c[1]]}`);b.setAttribute('aria-pressed',String(picker.selected.includes(c)));$('deck').append(b);}
  $('confirm-cards').disabled=picker.selected.length!==picker.count;
}
$('confirm-cards').onclick=()=>task(async()=>{
  $('card-error').hidden=true;$('confirm-cards').disabled=true;
  try{
    if(picker.count===3){await startSolve(picker.selected);}
    else{
      const next=branchPath(picker.context.path,picker.prefix,'deal:'+picker.selected[0]);
      const result=await api(`/api/jobs/${picker.context.job.id}/node`,{path:next.focus});
      ctx=picker.context;ctx.path=next.path;if(next.changed)ctx.end=result;node=result;phase='postflop';render(true);
    }
    $('card-dialog').close();
  }finally{$('confirm-cards').disabled=picker.selected.length!==picker.count;}
},'card-error');
$('card-close').onclick=()=>$('card-dialog').close();
$('card-dialog').addEventListener('cancel',e=>{if(loading)e.preventDefault();});
$('clear-cards').onclick=()=>{picker.selected=[];renderPicker();};
$('random-cards').onclick=()=>{const available=deck.filter(c=>!picker.blocked.includes(c));picker.selected=[];while(picker.selected.length<picker.count){const index=Math.floor(Math.random()*available.length);picker.selected.push(available.splice(index,1)[0]);}renderPicker();};
$('range-grid').onmouseleave=()=>{hovered=null;renderHand(pinned);};
$('range-grid').onfocusout=e=>{if(!$('range-grid').contains(e.relatedTarget)){hovered=null;renderHand(pinned);}};
$('unpin').onclick=()=>{pinned=null;renderSelection();};
for(const b of document.querySelectorAll('[data-view]'))b.onclick=()=>{view=b.dataset.view;document.querySelectorAll('[data-view]').forEach(t=>t.setAttribute('aria-selected',String(t===b)));render();};
$('new-hand').onclick=()=>task(()=>resetHand());
$('settings-open').onclick=()=>{restoreSettings();$('settings-error').hidden=true;$('settings-dialog').showModal();};
$('settings-close').onclick=()=>$('settings-dialog').close();
$('stop-mode').onchange=updateSettingsFields;$('mode').onchange=updateSettingsFields;$('chance-mode').onchange=updateSettingsFields;
$('backend').onchange=()=>{if($('backend').value==='deep_cfr')$('stop-mode').value='time';updateSettingsFields();};
$('reference-preset').onclick=()=>{
  $('backend').value='deep_cfr';$('stop-mode').value='time';
  $('preflop-input').value='raise:2.5 / call';$('flop-input').value='As Kd 7c';
  updateSettingsFields();$('time-limit').focus();
};
async function validateConfig(candidate){await api('/api/settings/validate',candidate);}
$('settings-form').onsubmit=e=>{e.preventDefault();task(async()=>{
  const candidate=readConfig();await validateConfig(candidate);
  config=candidate;
  $('settings-dialog').close();render();
},'settings-error');};
$('load-spot').onclick=()=>task(async()=>{
  const candidate=readConfig();await validateConfig(candidate);
  const history=$('preflop-input').value.split('/').map(s=>s.trim()).filter(Boolean),pre=await preRequest(history);
  if(!pre.pending)throw new Error('Use a complete preflop line ending in a call or check, with chips remaining.');
  const text=$('flop-input').value.replace(/\s+/g,'');if(!/^(?:[2-9TJQKA][cdhs]){3}$/i.test(text))throw new Error('Enter three cards, such as As Kd 7c.');
  const board=text.match(/.{2}/g).map(c=>c[0].toUpperCase()+c[1].toLowerCase());
  await startSolve(board,candidate,pre.path,pre);$('settings-dialog').close();
},'settings-error');
$('stop').onclick=()=>task(async()=>{await api(`/api/jobs/${ctx.job.id}/stop`,{});});
$('resume').onclick=()=>task(async()=>{const {settings,...budget}=config;ctx.job=await api(`/api/jobs/${ctx.job.id}/resume`,{...budget,max_nodes:settings.max_nodes});controls();poll(ctx.job.id);});
$('refine').onclick=()=>task(async()=>{
  const parent=ctx,trail=[...parent.baseTrail,...node.timeline.filter(p=>p.kind!=='decision'||!samePath(p.path,node.path)).map(p=>({...p,jobId:parent.job.id}))];
  const job=await api(`/api/jobs/${parent.job.id}/refine`,{...config,path:node.path});
  const board=[...parent.board],rootBoard=[...node.board];saveContext(job,board,trail,rootBoard);node=null;render(true);poll(job.id);
});
function download(data,name){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=el('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('export').onclick=()=>task(async()=>download(await api(`/api/jobs/${ctx.job.id}/export`,{path:node.path}),'heads-up-continuation.json'));
$('compare').onclick=()=>task(async()=>{
  const target=ctx;
  comparison=await api(`/api/jobs/${target.job.id}/comparison`,{});
  $('comparison-note').textContent=comparison.notice;
  const rows=$('comparison-rows');rows.replaceChildren();
  for(const entry of comparison.nodes){
    const n=entry.model,r=entry.reference,tr=el('tr');
    tr.append(el('th','',entry.label));
    const ours=el('td'),reference=el('td'),frequencies=el('td'),inspect=el('td');
    for(const [cell,data,selected] of [[ours,n,entry.model_selected],[reference,r,r.selected]]){
      cell.append(el('div','',selected?selected.replace(':',' '):'Awaiting action'),
        el('small','',`Pot ${fmt(data.pot)} BB · stacks ${data.stacks.map(fmt).join(' / ')}`));
    }
    if(n.aggregate.length)n.actions.forEach((a,i)=>frequencies.append(el('div','',`${shortAction(a)}: ${pct(n.aggregate[i])}`)));
    else frequencies.append(el('div','','No completed strategy yet'));
    frequencies.append(el('small','',`${pct(n.coverage)} ${n.neural?'direct samples':'with samples'}`));
    if(n.fallback_used)frequencies.append(el('small','','Earlier ranges use an untrained uniform fallback'));
    inspect.append(button('View','quiet',()=>task(async()=>{
      const river=comparison.nodes.at(-1).model;
      target.path=[...comparison.path];target.end=river;
      await inspectPost(target,n.path);$('comparison-dialog').close();
    })));
    tr.append(ours,reference,frequencies,inspect);rows.append(tr);
  }
  $('comparison-dialog').showModal();
});
$('comparison-close').onclick=()=>$('comparison-dialog').close();
$('comparison-export').onclick=()=>{if(comparison)download(comparison,'gto-wizard-line-comparison.json');};
$('download-chart').onclick=()=>task(async()=>download(await api('/api/builtin-chart'),'gto-wizard-preflop.json'),'settings-error');
updateSettingsFields();config=readConfig();task(()=>resetHand());
