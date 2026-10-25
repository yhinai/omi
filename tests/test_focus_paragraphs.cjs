'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {JSDOM}=require('jsdom');
const html=fs.readFileSync(require('node:path').join(__dirname,'../omi_focus.html'),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
function page(){
  const dom=new JSDOM(html,{url:'http://127.0.0.1:8788/focus',runScripts:'outside-only',pretendToBeVisual:true});
  const w=dom.window;
  w.matchMedia=()=>({matches:false,addEventListener(){}});
  w.requestAnimationFrame=()=>1;w.cancelAnimationFrame=()=>{};
  w.scrollBy=w.scrollTo=()=>{};
  w.fetch=async()=>({ok:false,status:404});
  w.WebSocket=class{static OPEN=1;static CONNECTING=0;constructor(){this.readyState=0;}close(){this.readyState=3;}};
  vm.runInContext(script,dom.getInternalVMContext());
  return {w,run:code=>vm.runInContext(code,dom.getInternalVMContext()),close:()=>w.close()};
}
const start=new Date(2026,9,24,16,42,8).getTime()/1000;
function phrase(id,offset=0,extra={}){return {type:'transcript',event_id:id,session_id:1,text:'A completed thought.',audio_start_at:start+offset,audio_end_at:start+offset+4,...extra};}
function add(p,event){p.run(`add(${JSON.stringify(event)})`);}
const paragraphs=p=>[...p.w.document.querySelectorAll('#words .transcript-paragraph')];
test('one faint HH:MM:SS stamp belongs to the first phrase, not every packet',()=>{const p=page();try{add(p,phrase('a'));add(p,phrase('b',4.5));const rows=paragraphs(p);assert.equal(rows.length,1);const times=rows[0].querySelectorAll('.paragraph-stamp time');assert.equal(times.length,1);assert.equal(times[0].textContent,'16:42:08');assert.equal(times[0].dateTime,new Date(start*1000).toISOString());assert.equal(rows[0].querySelector('.chunk').textContent,'A completed thought.');assert.equal(rows[0].querySelectorAll('.chunk').length,2);}finally{p.close();}});
test('a two-second speech pause opens a new paragraph; long speech is not mistaken for silence',()=>{const p=page();try{add(p,phrase('a',0,{audio_end_at:start+20}));add(p,phrase('b',20.5));assert.equal(paragraphs(p).length,1);add(p,phrase('c',26.5));assert.equal(paragraphs(p).length,2);assert.equal(paragraphs(p)[1].querySelector('time').textContent,'16:42:34');}finally{p.close();}});
test('final processing time is only a conservative pause fallback, not a packet boundary',()=>{const p=page();try{const first=phrase('a',0,{processed_at:start+8});delete first.audio_end_at;add(p,first);add(p,phrase('b',9));assert.equal(paragraphs(p).length,1);add(p,phrase('c',15));assert.equal(paragraphs(p).length,2);}finally{p.close();}});
test('speaker, known topic and session changes create explicit paragraph boundaries',()=>{const p=page();try{add(p,phrase('a',0,{speaker_id:'one'}));add(p,phrase('b',1,{speaker_id:'two'}));add(p,phrase('c',2,{speaker_id:'two',topic_changed:true}));add(p,phrase('d',3,{session_id:2}));assert.equal(paragraphs(p).length,4);}finally{p.close();}});
test('length cap waits for a complete sentence instead of splitting an unfinished thought',()=>{const p=page();try{add(p,phrase('a',0,{text:Array(105).fill('word').join(' ')}));add(p,phrase('b',1,{text:'and the sentence ends here.'}));assert.equal(paragraphs(p).length,1);add(p,phrase('c',2));assert.equal(paragraphs(p).length,2);}finally{p.close();}});
test('partial revisions never create extra paragraph timestamps or saved text',()=>{const p=page();try{add(p,phrase('a'));p.run('renderPartial({type:"partial",utterance_id:"u",revision:1,text:"The next",session_id:1})');const draft=p.w.document.querySelector('.live-draft');p.run('renderPartial({type:"partial",utterance_id:"u",revision:2,text:"The next thought.",session_id:1})');assert.equal(p.w.document.querySelector('.live-draft'),draft);assert.equal(p.w.document.querySelectorAll('#words time').length,1);assert.equal(p.run('entries.length'),1);add(p,phrase('final',4.5,{utterance_id:'u',text:'The next thought.'}));assert.equal(p.w.document.querySelector('.live-draft'),null);assert.equal(paragraphs(p).length,1);assert.equal(p.w.document.querySelectorAll('#words time').length,1);assert.equal(p.w.document.querySelectorAll('#words .pending').length,0);}finally{p.close();}});
test('empty live drafts and expired history do not leave orphan timestamp rows',()=>{const p=page();try{p.run(`renderPartial(${JSON.stringify({type:'partial',utterance_id:'empty',revision:1,text:'Draft',session_id:1,audio_start_at:start})})`);p.run('clearDraft("empty")');assert.equal(paragraphs(p).length,0);for(let i=0;i<165;i++)add(p,phrase(String(i),i*10));assert.equal(paragraphs(p).length,160);assert.equal(p.w.document.querySelectorAll('#words time').length,160);assert(paragraphs(p).every(row=>row.querySelector('.chunk')));}finally{p.close();}});
