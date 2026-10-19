import test from 'node:test';
import assert from 'node:assert/strict';
import {branchPath,handCombos,comboKey,isBlocked,chooseCard,deck,groupRows,actionColors,trainingLimit} from '../heads_up_gto/static/model.mjs';

test('training controls submit only the selected limit and ignore the inactive field',()=>{
  assert.deepEqual(trainingLimit('time',NaN,60),{time_limit_seconds:60});
  assert.deepEqual(trainingLimit('iterations',200001,NaN),{iterations:200001});
  for(const seconds of [0,-1,Infinity,NaN])assert.throws(()=>trainingLimit('time',2000,seconds));
  for(const iterations of [0,9,10.5,Infinity,NaN])assert.throws(()=>trainingLimit('iterations',iterations,60));
});

test('revisiting the same action preserves future streets; changing it forks the line',()=>{
  const path=['check','check','deal:Ts','bet:2.5','call'];
  assert.deepEqual(branchPath(path,[],'check'),{path,focus:['check'],changed:false});
  assert.deepEqual(branchPath(path,['check','check'],'deal:Qh'),{path:['check','check','deal:Qh'],focus:['check','check','deal:Qh'],changed:true});
  assert.deepEqual(branchPath(path,[],'bet:2.5').path,['bet:2.5']);
  assert.throws(()=>branchPath(path,['bet:1'],'call'));
});
test('hand details enumerate the correct distinct suit combinations and blockers',()=>{
  assert.equal(handCombos('AA').length,6);assert.equal(handCombos('AKs').length,4);assert.equal(handCombos('AKo').length,12);
  for(const name of ['AA','AKs','AKo'])assert.equal(new Set(handCombos(name).map(comboKey)).size,handCombos(name).length);
  assert.equal(handCombos('AA').filter(h=>!isBlocked(h,['As'])).length,3);
  assert.equal(handCombos('AKs').filter(h=>!isBlocked(h,['As','Kd'])).length,2);
  assert.equal(comboKey('AhAs'),comboKey('AsAh'));
});
test('card picker prevents duplicates and dealt cards and supports deselect/replace',()=>{
  assert.equal(deck.length,52);assert.equal(new Set(deck).size,52);
  assert.deepEqual(chooseCard(['As'],'As',3,[]),[]);
  assert.deepEqual(chooseCard(['As'],'Kd',3,[]),['As','Kd']);
  assert.deepEqual(chooseCard(['As'],'Kd',1,[]),['Kd']);
  assert.deepEqual(chooseCard(['As'],'Kd',3,['Kd']),['As']);
  assert.deepEqual(chooseCard(['As'],'Xx',3,[]),['As']);
});
test('range summaries weight frequencies by compatible mass and retain unknown mass',()=>{
  const actions=[{kind:'check'},{kind:'bet'}],rows=[{class:'AA',mass:.3,weight:1,probabilities:[1,0]},{class:'AA',mass:.1,weight:.5,probabilities:[0,1]},{class:'AA',mass:.2,weight:.3,probabilities:null}];
  const g=groupRows(rows,actions).get('AA');assert.equal(g.weight,1.8);assert.equal(g.covered,.4);assert.deepEqual(g.totals,[.3,.1]);assert.ok(Math.abs(g.mass-.6)<1e-9);
});
test('action colors follow meaning even when action order changes',()=>{
  assert.deepEqual(actionColors([{kind:'call'},{kind:'fold'},{kind:'raise',label:'Raise 26'},{kind:'raise',label:'All-in 100'}]),['#58b568','#4387bd','#ed484b','#7b252c']);
});
