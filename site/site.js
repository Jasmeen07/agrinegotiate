(function(){
'use strict';
var D=window.D, NS='http://www.w3.org/2000/svg';
var css=getComputedStyle(document.documentElement);
function cv(n){return css.getPropertyValue(n).trim();}
var INK=cv('--ink'),INK2=cv('--ink2'),INK3=cv('--ink3'),RULE=cv('--rule'),WHEAT=cv('--wheat'),LEAF=cv('--leaf'),CLAY=cv('--clay');
function el(n,a,txt){var e=document.createElementNS(NS,n);for(var k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;return e;}
function add(p,n,a,t){var e=el(n,a,t);p.appendChild(e);return e;}
function clear(n){while(n.firstChild)n.removeChild(n.firstChild);}
function $(id){return document.getElementById(id);}
function f2(x){if(x==null)return 'n/a';var s=x.toFixed(2);return s==='-0.00'?'0.00':s;}

/* ---------- Fig 1: the problem ---------- */
(function(){
  var s=$('fig-problem'); var H=170;
  add(s,'line',{x1:0,y1:H,x2:900,y2:H,stroke:INK,'stroke-width':1.5});
  for(var x=-200;x<=1100;x+=100){add(s,'path',{d:'M'+x+' 270 Q '+(x+(450-x)*0.45)+' 210 450 '+H,fill:'none',stroke:INK3,'stroke-width':1,opacity:.45});}
  add(s,'circle',{cx:300,cy:62,r:22,fill:'none',stroke:WHEAT,'stroke-width':2});
  function person(cx,hat){
    add(s,'circle',{cx:cx,cy:112,r:11,fill:INK});
    add(s,'path',{d:'M'+(cx-17)+' 168 L'+(cx-9)+' 128 L'+(cx+9)+' 128 L'+(cx+17)+' 168 Z',fill:INK});
    if(hat)add(s,'rect',{x:cx-14,y:98,width:28,height:5,fill:INK});
  }
  person(150,false); person(750,true);
  add(s,'rect',{x:178,y:148,width:26,height:20,fill:'none',stroke:INK,'stroke-width':1.5});
  add(s,'text',{x:191,y:162,'text-anchor':'middle','font-size':11,fill:INK},'sack');
  add(s,'rect',{x:704,y:150,width:20,height:18,fill:'none',stroke:INK,'stroke-width':1.5});
  add(s,'text',{x:150,y:196,'text-anchor':'middle','font-size':14,fill:INK2},'Farmer');
  add(s,'text',{x:150,y:214,'text-anchor':'middle','font-size':12,fill:INK3},'fallback: unknown to buyer');
  add(s,'text',{x:750,y:196,'text-anchor':'middle','font-size':14,fill:INK2},'Buyer');
  add(s,'text',{x:750,y:214,'text-anchor':'middle','font-size':12,fill:INK3},'value: unknown to farmer');
  add(s,'path',{d:'M215 108 Q450 20 690 108',fill:'none',stroke:INK,'stroke-width':1.5,'stroke-dasharray':'6 7'});
  [[6,0],[10,0],[14,1],[18,1]].forEach(function(b,i){add(s,'rect',{x:430+i*10,y:92-b[0],width:6,height:b[0],fill:b[1]?'none':INK,stroke:INK,'stroke-width':1});});
  add(s,'text',{x:450,y:112,'text-anchor':'middle','font-size':13,fill:INK2},'signal: half the time');
  add(s,'text',{x:450,y:248,'text-anchor':'middle','font-size':16,fill:INK,'font-style':'italic','font-family':'Fraunces,Georgia,serif'},'Where between the two private prices does the deal land?');
})();

/* ---------- Fig 2: worked example ---------- */
(function(){
  var s=$('fig-example'), L=70,R=790,T=24,B=260;
  function X(t){return L+t/10*(R-L);}
  function Y(p){return B-(p-700)/700*(B-T);}
  add(s,'rect',{x:L,y:Y(1300),width:R-L,height:Y(800)-Y(1300),fill:WHEAT,opacity:.16});
  [[800,'Farmer fallback 800'],[1300,'Buyer value 1,300']].forEach(function(a){
    add(s,'line',{x1:L,y1:Y(a[0]),x2:R,y2:Y(a[0]),stroke:INK,'stroke-width':1,'stroke-dasharray':'5 5'});
    add(s,'text',{x:R,y:Y(a[0])+(a[0]===800?16:-6),'text-anchor':'end','font-size':13,fill:INK2},a[1]);
  });
  add(s,'line',{x1:L,y1:Y(1050),x2:R,y2:Y(1050),stroke:LEAF,'stroke-width':1.5});
  add(s,'text',{x:L+6,y:Y(1050)-6,'font-size':13,fill:LEAF},'Even split: 1,050');
  add(s,'line',{x1:L,y1:Y(930),x2:R,y2:Y(930),stroke:CLAY,'stroke-width':1,'stroke-dasharray':'2 4'});
  add(s,'text',{x:R,y:Y(930)-6,'text-anchor':'end','font-size':13,fill:CLAY},'Farmer share 26%: 930');
  var F=[1250,1200,1150,1100,1070,1050], Bu=[850,900,950,1000,1030,1050];
  var pf='',pb='';
  F.forEach(function(p,i){var x=X(i*1.6+.2);pf+=(i?'L':'M')+x+' '+Y(p)+' ';add(s,'circle',{cx:x,cy:Y(p),r:5,fill:INK});});
  Bu.forEach(function(p,i){var x=X(i*1.6+.2);pb+=(i?'L':'M')+x+' '+Y(p)+' ';add(s,'circle',{cx:x,cy:Y(p),r:5,fill:'var(--paper)',stroke:INK,'stroke-width':1.8});});
  add(s,'path',{d:pf,fill:'none',stroke:INK,'stroke-width':1.2,opacity:.6});
  add(s,'path',{d:pb,fill:'none',stroke:INK,'stroke-width':1.2,opacity:.6,'stroke-dasharray':'4 3'});
  var dx=X(8.2),dy=Y(1050);
  add(s,'path',{d:'M'+dx+' '+(dy-9)+' L'+(dx+9)+' '+dy+' L'+dx+' '+(dy+9)+' L'+(dx-9)+' '+dy+' Z',fill:WHEAT,stroke:INK});
  add(s,'text',{x:dx,y:dy-16,'text-anchor':'middle','font-size':13,fill:INK},'Deal');
  add(s,'text',{x:X(.2)+10,y:Y(1250)+4,'font-size':12,fill:INK2},'farmer asks');
  add(s,'text',{x:X(.2)+10,y:Y(850)+4,'font-size':12,fill:INK2},'buyer bids');
  add(s,'text',{x:L-8,y:Y(1300)+4,'text-anchor':'end','font-size':12,fill:INK3},'1,300');
  add(s,'text',{x:L-8,y:Y(800)+4,'text-anchor':'end','font-size':12,fill:INK3},'800');
  add(s,'text',{x:(L+R)/2,y:B+30,'text-anchor':'middle','font-size':12,fill:INK3},'time, left to right');
})();

/* ---------- episode renderer (live + replay) ---------- */
function drawEp(svg,E){
  clear(svg);
  var W=740,L=48,R=720,T=14,B=262,P0=0.2,P1=1.2;
  function X(t){return L+(t+.5)/E.T*(R-L);}
  function Y(p){return B-(p-P0)/(P1-P0)*(B-T);}
  add(svg,'rect',{x:L,y:Y(E.v),width:R-L,height:Math.max(0,Y(E.fb)-Y(E.v)),fill:WHEAT,opacity:.17});
  [0.2,0.4,0.6,0.8,1.0,1.2].forEach(function(p){
    add(svg,'line',{x1:L,y1:Y(p),x2:R,y2:Y(p),stroke:RULE,'stroke-width':.7});
    add(svg,'text',{x:L-6,y:Y(p)+4,'text-anchor':'end','font-size':11,fill:INK3},p.toFixed(1));
  });
  add(svg,'line',{x1:L,y1:Y(E.fb),x2:R,y2:Y(E.fb),stroke:INK,'stroke-dasharray':'5 4','stroke-width':1});
  add(svg,'text',{x:R,y:Y(E.fb)+14,'text-anchor':'end','font-size':11.5,fill:INK2},'farmer fallback');
  add(svg,'line',{x1:L,y1:Y(E.v),x2:R,y2:Y(E.v),stroke:INK,'stroke-dasharray':'5 4','stroke-width':1});
  add(svg,'text',{x:R,y:Y(E.v)-5,'text-anchor':'end','font-size':11.5,fill:INK2},'buyer value');
  add(svg,'line',{x1:L,y1:Y(E.nash),x2:R,y2:Y(E.nash),stroke:LEAF,'stroke-width':1});
  add(svg,'text',{x:L+4,y:Y(E.nash)-4,'font-size':11.5,fill:LEAF},'even split');
  if(E.floor){add(svg,'line',{x1:L,y1:Y(E.floor),x2:R,y2:Y(E.floor),stroke:CLAY,'stroke-width':1.2});add(svg,'text',{x:L+4,y:Y(E.floor)+13,'font-size':11.5,fill:CLAY},'price floor');}
  [0,1].forEach(function(side){
    var pts=E.sends.filter(function(s){return s[1]===side&&s[0]<=E.now;}),d='';
    pts.forEach(function(s,i){
      var nx=i+1<pts.length?pts[i+1][0]:Math.min(E.now,E.deal?E.deal[0]:E.now);
      d+=(i?'M':'M')+X(s[0])+' '+Y(s[2])+' H'+X(Math.max(s[0],nx))+' ';
    });
    if(d)add(svg,'path',{d:d,fill:'none',stroke:INK,'stroke-width':1.2,opacity:.55,'stroke-dasharray':side?'4 3':''});
    pts.forEach(function(s){
      add(svg,'circle',{cx:X(s[0]),cy:Y(s[2]),r:4.2,fill:side?'var(--paper)':INK,stroke:INK,'stroke-width':1.6,opacity:s[3]?.35:1});
    });
  });
  if(E.deal&&E.deal[0]<=E.now){
    var x=X(E.deal[0]),y=Y(E.deal[1]);
    add(svg,'path',{d:'M'+x+' '+(y-9)+' L'+(x+9)+' '+y+' L'+x+' '+(y+9)+' L'+(x-9)+' '+y+' Z',fill:WHEAT,stroke:INK,'stroke-width':1.2});
    add(svg,'text',{x:x,y:y-14,'text-anchor':'middle','font-size':12,fill:INK,'font-weight':600},E.deal[1].toFixed(2));
  }
  if(E.now<E.T)add(svg,'line',{x1:X(E.now),y1:T,x2:X(E.now),y2:B,stroke:WHEAT,'stroke-width':1.2,opacity:.8});
  [0,5,10,15,20,25,30].forEach(function(t){if(t<=E.T)add(svg,'text',{x:X(t),y:B+14,'text-anchor':'middle','font-size':11,fill:INK3},t);});
  if(E.links){
    add(svg,'text',{x:L-6,y:B+34,'text-anchor':'end','font-size':11,fill:INK3},'link');
    for(var t=0;t<E.T;t++){
      var on=E.links[t]; if(t>E.now||on==null)continue;
      add(svg,'rect',{x:X(t)-(R-L)/E.T/2+1,y:B+24,width:(R-L)/E.T-2,height:9,fill:on?LEAF:CLAY,opacity:on?.55:.85});
    }
  }
}

/* ---------- simulator ---------- */
var GRID=[];for(var k=0;k<=20;k++)GRID.push(+(0.2+k*0.05).toFixed(2));
function snap(p){var b=GRID[0];GRID.forEach(function(g){if(Math.abs(g-p)<Math.abs(b-p))b=g;});return b;}
function rng(seed){var a=seed>>>0;return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
var STR={'Linear':{beta:1},'Boulware (holds out)':{beta:.2},'Conceder (gives in)':{beta:5},'Hardball':{hard:1},'Random sensible offers':{zic:1}};
var T=30,TTL=6,REF=1.1,KAPPA=0.6;
var S=null,timer=null;

function fillSel(id,extra){var s=$(id);Object.keys(STR).forEach(function(k){var o=document.createElement('option');o.value=k;o.textContent=k;s.appendChild(o);});
  var o=document.createElement('option');o.value='YOU';o.textContent='You';s.appendChild(o);}
fillSel('sFarmer');fillSel('sBuyer');
$('sFarmer').value='Linear';$('sBuyer').value='Linear';

var seedCounter=7;
function newState(){
  var duty=+$('sLink').value, f=+$('sF').value, v=+$('sV').value, r=rng(seedCounter*7919+13);
  var S={t:0,f:f,v:v,duty:duty,rule:$('sRule').value,fs:$('sFarmer').value,bs:$('sBuyer').value,rand:r,
    link:[],on:true,fo:null,bo:null,fv:null,bv:null,sends:[],deal:null,log:[],done:false,pend:null,held:{0:null,1:null},linkHist:[]};
  S.floor=S.rule==='floor'?KAPPA*REF:null;
  S.nash=(f+v)/2;
  S.on=r()<duty;S.linkHist[0]=S.on;
  return S;
}
function tgt(side,st,t){
  var spec=STR[side?st.bs:st.fs], f=st.f, v=st.v;
  var start=side?0.2:1.2, reserve=side?v:f;
  if(spec.hard){return side?snap(f+0.05):snap(v-0.05);}
  if(spec.zic){var a=side?0.2:f, b=side?v:1.2;return snap(a+st.rand()*(b-a));}
  var a2=Math.min(1,t/(T-2)); a2=Math.pow(a2,1/spec.beta);
  return snap(start+a2*(reserve-start));
}
function ok(side,p,st){return side?p<=st.v+1e-9:p>=st.f-1e-9;}
function log(st,msg,cls){st.log.push({t:st.t,m:msg,c:cls||''});}
function humanSide(){return $('sFarmer').value==='YOU'?0:($('sBuyer').value==='YOU'?1:null);}
function needsInput(st){
  var hs=st.fs==='YOU'?0:(st.bs==='YOU'?1:null);
  if(hs==null||st.done)return false;
  if(hs===0&&!st.on)return false;
  return st.pend==null;
}
function tick(st){
  if(st.done)return;
  var t=st.t;
  if(t>0&&st.duty<1){var r=st.rand();
    if(st.on){var pOff=0.5*(1-st.duty)/st.duty;if(r<pOff)st.on=false;}else if(r<0.5)st.on=true;}
  if(t>0&&st.duty>=1)st.on=true;
  st.linkHist[t]=st.on;
  // deliveries
  if(st.fo&&st.bv!==st.fo&&t>=st.fo.t+1&&t-st.fo.t<=TTL){st.bv=st.fo;}
  if(st.bv&&t-st.bv.t>TTL){st.bv=null;}
  if(st.bo&&t>=st.bo.t+1&&t-st.bo.t<=TTL&&st.on&&st.fv!==st.bo){st.fv=st.bo;log(st,'Farmer receives buyer offer '+st.bo.p.toFixed(2));}
  if(st.fv&&t-st.fv.t>TTL){log(st,'Buyer offer '+st.fv.p.toFixed(2)+' expired unread','bad');st.fv=null;if(st.bo&&t-st.bo.t>TTL)st.bo=null;}
  if(!st.on&&t>0&&st.linkHist[t-1])log(st,'Farmer link goes down','bad');
  if(st.on&&t>0&&!st.linkHist[t-1])log(st,'Farmer link is back');
  var acts=[null,null];
  [0,1].forEach(function(side){
    if(side===0&&!st.on)return;
    var strat=side?st.bs:st.fs, view=side?st.bv:st.fv;
    if(strat==='YOU'){acts[side]=st.pend||{k:'wait'};return;}
    var target=tgt(side,st,t);
    if(side===1&&st.floor&&target<st.floor)target=Math.min(snap(st.floor+0.0),st.v);
    var good=view&&(side?view.p<=target+1e-9:view.p>=target-1e-9)&&ok(side,view.p,st);
    if(good){acts[side]={k:'accept'};return;}
    var spec=STR[strat], own=side?st.bo:st.fo;
    if(spec.zic&&own&&t-own.t<3){acts[side]={k:'wait'};return;}
    if(own&&Math.abs(own.p-target)<1e-6&&t-own.t<TTL-1){acts[side]={k:'wait'};return;}
    if(side===1&&st.floor&&st.floor>st.v+1e-9){acts[side]={k:'wait'};return;}
    acts[side]={k:'offer',p:target};
  });
  st.pend=null;
  [0,1].forEach(function(side){
    var a=acts[side]; if(!a||st.done)return;
    var view=side?st.bv:st.fv, name=side?'Buyer':'Farmer';
    if(a.k==='offer'){
      var p=a.p; if(side===1&&st.floor&&p<st.floor)p=snap(st.floor);
      var o={p:p,t:t,side:side};
      if(side)st.bo=o;else st.fo=o;
      st.sends.push([t,side,p,0]);
      log(st,name+' offers '+p.toFixed(2));
    }else if(a.k==='accept'&&view&&t-view.t<=TTL){
      if(!ok(side,view.p,st))return;
      if(view.p<(st.floor||0)-1e-9)return;
      st.deal=[t,view.p,0];
      var tp=st.f+0.4*(st.v-st.f), fee=st.rule==='fee'?Math.max(0,tp-view.p):0;
      st.deal[2]=fee;
      log(st,name+' accepts '+view.p.toFixed(2),'deal');
      st.done=true;
    }else if(a.k==='walk'){
      log(st,name+' walks away','bad');st.done=true;st.walked=true;
    }
  });
  st.t=t+1;
  if(!st.done&&st.t>=T){st.done=true;log(st,'Deadline: no deal','bad');}
}
function adv(st,once){
  var guard=0;
  while(!st.done&&guard++<40){
    if(needsInput(st))break;
    tick(st);
    if(once)break;
  }
}
function readout(st){
  var r=$('readout'),h='';
  if(st.deal){
    var fg=st.deal[1]-st.f,bg=st.v-st.deal[1],gain=st.v-st.f;
    h='<span>Price <b>'+st.deal[1].toFixed(2)+'</b></span><span>Farmer share <b>'+f2(fg/gain)+'</b></span><span>Farmer gains <b>'+f2(fg)+'</b></span><span>Buyer gains <b>'+f2(bg)+'</b></span><span>Efficiency <b>1.00</b></span>';
    if(st.rule==='fee')h+='<span>Fee charged to buyer <b>'+f2(st.deal[2])+'</b></span>';
  }else if(st.done){
    h='<span>No deal. Farmer keeps fallback <b>'+st.f.toFixed(2)+'</b></span><span>Efficiency <b>0.00</b></span><span>Farmer share <b>n/a</b></span>';
  }else{h='<span>Even split would be <b>'+st.nash.toFixed(2)+'</b></span><span>Gap to share <b>'+(st.v-st.f).toFixed(2)+'</b></span>';}
  r.innerHTML=h;
}
function renderLog(st){
  var d=$('log'),h='';
  st.log.slice(-60).forEach(function(l){h+='<div class="'+l.c+'">t='+(l.t<10?' ':'')+l.t+'  '+l.m+'</div>';});
  d.innerHTML=h;d.scrollTop=d.scrollHeight;
}
function renderHuman(st){
  var hb=$('human'),hs=st.fs==='YOU'?0:(st.bs==='YOU'?1:null);
  if(hs==null){hb.style.display='none';return;}
  hb.style.display='block';
  var name=hs?'buyer':'farmer', view=hs?st.bv:st.fv;
  if(st.done){hb.innerHTML='You played the '+name+'. Press New draw to try again.';return;}
  if(hs===0&&!st.on){hb.innerHTML='You are the farmer. <b>Your phone link is down.</b> Offers that arrive now may expire before you see them.<div class="row"><button id="hCont">Continue</button></div>';
    $('hCont').onclick=function(){adv(st,true);paint(st);};return;}
  var def=hs?0.5:1.0;
  var acceptable=view&&((hs&&view.p<=st.v+1e-9)||(!hs&&view.p>=st.f-1e-9))&&(!st.floor||view.p>=st.floor-1e-9);
  hb.innerHTML='You are the <b>'+name+'</b>. '+(hs?'Your value is '+st.v.toFixed(2)+'.':'Your fallback is '+st.f.toFixed(2)+'.')+
   ' Their open offer: <b>'+(view?view.p.toFixed(2):'none yet')+'</b>.<div class="row"><input type="range" id="hP" min="0.2" max="1.2" step="0.05" value="'+def+'"><span class="num" id="hPv">'+def.toFixed(2)+'</span>'+
   '<button class="go" id="hOffer">Send offer</button><button id="hAcc" '+(acceptable?'':'disabled')+'>Accept theirs</button><button id="hWait">Wait</button><button id="hWalk">Walk away</button></div>';
  $('hP').oninput=function(){$('hPv').textContent=(+this.value).toFixed(2);};
  function go(a){st.pend=a;adv(st,false);paint(st);}
  $('hOffer').onclick=function(){go({k:'offer',p:snap(+$('hP').value)});};
  $('hAcc').onclick=function(){go({k:'accept'});};
  $('hWait').onclick=function(){go({k:'wait'});};
  $('hWalk').onclick=function(){go({k:'walk'});};
}
function paint(st){
  drawEp($('simsvg'),{T:T,f:st.f,v:st.v,fb:st.f,nash:st.nash,sends:st.sends,deal:st.deal,now:st.deal?st.deal[0]:Math.max(0,st.t-1),links:st.linkHist,floor:st.floor});
  $('vT').textContent=st.t;
  readout(st);renderLog(st);renderHuman(st);
  $('bPlay').textContent=timer?'Pause':'Play';
  $('bStep').disabled=!!st.done||needsInput(st);
  if(st.done&&timer){clearInterval(timer);timer=null;$('bPlay').textContent='Play';}
}
function reset(newSeed){
  if(timer){clearInterval(timer);timer=null;}
  if(newSeed)seedCounter++;
  S=newState();paint(S);
}
function play(){
  if(timer){clearInterval(timer);timer=null;paint(S);return;}
  if(S.done)reset(true);
  timer=setInterval(function(){
    if(needsInput(S)){clearInterval(timer);timer=null;paint(S);return;}
    adv(S,true);paint(S);
  },330);
  paint(S);
}
$('bPlay').onclick=play;
$('bStep').onclick=function(){if(S.done)reset(true);adv(S,true);paint(S);};
$('bReset').onclick=function(){reset(true);};
['sFarmer','sBuyer','sLink','sRule','sF','sV'].forEach(function(id){$(id).onchange=function(){$('vF').textContent=(+$('sF').value).toFixed(2);$('vV').textContent=(+$('sV').value).toFixed(2);reset(false);};});
$('sF').oninput=function(){$('vF').textContent=(+this.value).toFixed(2);};
$('sV').oninput=function(){$('vV').textContent=(+this.value).toFixed(2);};
reset(false);

/* ---------- replay ---------- */
var RS=null,rt=0,rtimer=null;
(function(){
  var sel=$('rSel');
  D.R.forEach(function(r,i){var o=document.createElement('option');o.value=i;o.textContent=r.title;sel.appendChild(o);});
  function load(){
    if(rtimer){clearInterval(rtimer);rtimer=null;}
    RS=D.R[+sel.value];rt=0;$('rNote').textContent=RS.note+(RS.duty<1?' The farmer\'s link was up '+Math.round(RS.duty*100)+'% of the time.':'');
    rpaint();
  }
  function rpaint(){
    var d=RS.deals.length?RS.deals[0]:null;
    drawEp($('rsvg'),{T:RS.T,f:RS.fb,v:RS.v,fb:RS.fb,nash:RS.nash,sends:RS.sends,deal:d,now:rt});
    var h='';
    if(d&&d[0]<=rt){h='<span>Price <b>'+d[1].toFixed(2)+'</b></span><span>Efficiency <b>'+f2(RS.E)+'</b></span><span>Farmer gain capture <b>'+f2(RS.FGC)+'</b></span>'+(d[2]>0?'<span>Fee charged <b>'+f2(d[2])+'</b></span>':'');}
    else if(rt>=RS.T){h='<span>No deal</span>';}
    else h='<span>Even split <b>'+RS.nash.toFixed(2)+'</b></span><span>Time <b>'+rt+'</b> of '+RS.T+'</span>';
    $('rread').innerHTML=h;
    var end=d?d[0]:RS.T;
    $('rPlay').textContent=rtimer?'Pause':'Play';
    if(rt>=end&&rtimer){clearInterval(rtimer);rtimer=null;$('rPlay').textContent='Play';}
  }
  $('rPlay').onclick=function(){
    if(rtimer){clearInterval(rtimer);rtimer=null;rpaint();return;}
    var d=RS.deals.length?RS.deals[0]:null,end=d?d[0]:RS.T;
    if(rt>=end)rt=0;
    rtimer=setInterval(function(){rt++;rpaint();},380);rpaint();
  };
  $('rReset').onclick=function(){rt=0;if(rtimer){clearInterval(rtimer);rtimer=null;}rpaint();};
  sel.onchange=load;load();
})();
document.querySelectorAll('#simtabs button').forEach(function(b){b.onclick=function(){
  document.querySelectorAll('#simtabs button').forEach(function(x){x.classList.toggle('on',x===b);});
  $('tab-live').style.display=b.dataset.tab==='live'?'block':'none';
  $('tab-replay').style.display=b.dataset.tab==='replay'?'block':'none';
};});

/* ---------- table 1 ---------- */
(function(){
  var rows=D.S.filter(function(s){return s.scenario==='SYM'&&s.farmer===s.buyer;});
  var h='<table><thead><tr><th>Both sides play</th><th class="r">Efficiency</th><th class="r">Farmer share</th><th class="r">Deal rate</th></tr></thead><tbody>';
  rows.forEach(function(s){h+='<tr><td>'+s.farmer+'</td><td class="r">'+f2(s.E)+'</td><td class="r">'+f2(s.S_F)+'</td><td class="r">'+f2(s.agreement_rate)+'</td></tr>';});
  $('tbl-ref').innerHTML=h+'</tbody></table>';
})();

/* ---------- dot plots ---------- */
function dotplot(host,rows,o){
  var rh=28,LW=o.labelW||200,PW=o.plotW||300,W=LW+PW+95,H=rows.length*rh+44,top=10;
  var s=el('svg',{viewBox:'0 0 '+W+' '+H,role:'img','aria-label':o.label||'chart'});
  function X(v){return LW+(v-o.min)/(o.max-o.min)*PW;}
  o.ticks.forEach(function(t){add(s,'line',{x1:X(t),y1:top,x2:X(t),y2:H-30,stroke:RULE,'stroke-width':.7});add(s,'text',{x:X(t),y:H-14,'text-anchor':'middle','font-size':11.5,fill:INK3},t.toFixed(1));});
  (o.refs||[]).forEach(function(r){add(s,'line',{x1:X(r.x),y1:top,x2:X(r.x),y2:H-30,stroke:INK,'stroke-dasharray':'4 3','stroke-width':1});add(s,'text',{x:X(r.x)+3,y:top+8,'font-size':11,fill:INK2},r.label);});
  rows.forEach(function(r,i){
    var y=top+20+i*rh, c=r.color||INK;
    add(s,'text',{x:LW-10,y:y+4,'text-anchor':'end','font-size':12.5,fill:INK},r.label);
    if(r.ci)add(s,'line',{x1:X(r.ci[0]),y1:y,x2:X(r.ci[1]),y2:y,stroke:c,'stroke-width':5,opacity:.28,'stroke-linecap':'round'});
    var runs=(r.runs||[]).filter(function(v){return v!=null;});
    runs.forEach(function(v,j){add(s,'circle',{cx:X(v),cy:y+((j%3)-1)*3.2,r:3.3,fill:c,opacity:.85});});
    if(r.iqm!=null)add(s,'line',{x1:X(r.iqm),y1:y-9,x2:X(r.iqm),y2:y+9,stroke:INK,'stroke-width':2});
    add(s,'text',{x:LW+PW+12,y:y+4,'font-size':11.5,fill:INK3},(r.iqm!=null?f2(r.iqm)+'  ':'')+'n='+r.n);
    if(!runs.length)add(s,'text',{x:X(o.min)+6,y:y+4,'font-size':11.5,fill:CLAY},'no trade in any run');
  });
  host.appendChild(s);
}
function row(key,label,metric,color){var g=D.G[key];if(!g)return {label:label+' (missing)',n:0,runs:[]};var m=g[metric];return {label:label,runs:m.runs,iqm:m.iqm,ci:m.ci,n:g.n,color:color};}

var NOR=INK3;
dotplot($('fig-h1'),[
  row('x2_gate/SYM','Equal conditions','S_F',INK),
  row('x2_gate/DEADLINE','Farmer deadline earlier','S_F',INK),
  row('x2_gate/INFO','Noisy price signal','S_F',INK),
  row('x2_gate/PERISH','Perishable crop','S_F',CLAY),
  row('x2_gate/THIN3','Three farmers, one buyer','S_F',CLAY)
],{min:0,max:0.8,ticks:[0,.2,.4,.6,.8],refs:[{x:.5,label:'equal split'}],plotW:380,labelW:200,label:'Farmer share by setting'});

var fl=[['floor_kappa0.50','Floor 0.50'],['floor_kappa0.55','Floor 0.55'],['floor_kappa0.60','Floor 0.60'],['floor_kappa0.65','Floor 0.65']];
function h3rows(metric){
  var r=[row('x2_gate/PERISH','No rule','S_F'.replace('S_F',metric),NOR),
    row('x3_mechanism/fee_oracle_tau0.4','Fee, standard',metric,WHEAT),
    row('x3_mechanism/fee_oracle_tau0.4_eta0.25','Fee, faster update',metric,WHEAT),
    row('x3_mechanism/fee_fixed_lam1.0_tau0.4','Fee, fixed rate',metric,WHEAT)];
  fl.forEach(function(f){r.push(row('x3_mechanism/'+f[0],f[1],metric,LEAF));});
  return r;
}
dotplot($('fig-h3a'),h3rows('S_F'),{min:0,max:0.8,ticks:[0,.2,.4,.6,.8],refs:[{x:.4,label:'target'}],plotW:240,labelW:130,label:'Farmer share by rule'});
dotplot($('fig-h3b'),h3rows('E'),{min:0.4,max:1,ticks:[.4,.6,.8,1],plotW:240,labelW:130,label:'Efficiency by rule'});

function hetRows(m){return [
  row('x6_hetero/unconstrained','No rule',m,NOR),
  row('x6_hetero/fee_oracle_tau0.4_eta0.25','Fee, faster update',m,WHEAT),
  row('x6_hetero/fee_fixed_lam1.0_tau0.4','Fee, fixed rate',m,WHEAT),
  row('x6_hetero/floor_kappa0.50','Floor 0.50',m,LEAF),
  row('x6_hetero/floor_kappa0.55','Floor 0.55',m,LEAF),
  row('x6_hetero/floor_kappa0.60','Floor 0.60',m,LEAF)];}
dotplot($('fig-het'),hetRows('A'),{min:0,max:1,ticks:[0,.2,.4,.6,.8,1],plotW:200,labelW:130,label:'Deal rate when buyer values vary widely'});
dotplot($('fig-hetE'),hetRows('E'),{min:0,max:1,ticks:[0,.2,.4,.6,.8,1],plotW:200,labelW:130,label:'Efficiency when buyer values vary widely'});

dotplot($('fig-h5'),[
  row('x4_delay/unaware_duty1.0','Unaware, link always up','E',INK2),
  row('x4_delay/aware_duty1.0','Delay-aware, link always up','E',WHEAT),
  row('x4_delay/unaware_duty0.6','Unaware, link up 60%','E',INK2),
  row('x4_delay/aware_duty0.6','Delay-aware, link up 60%','E',WHEAT),
  row('x4_delay/unaware_duty0.4','Unaware, link up 40%','E',INK2),
  row('x4_delay/aware_duty0.4','Delay-aware, link up 40%','E',WHEAT)
],{min:0,max:1,ticks:[0,.2,.4,.6,.8,1],plotW:380,labelW:210,label:'Efficiency by link uptime'});


/* ---------- wind-swept field ---------- */
function field(cv,o){
  o=o||{};
  var ctx=cv.getContext('2d'),W,H,dpr,rows,motes,mx=-9999,mv=0,t=0,last=0,vis=true,
      reduce=window.matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches;
  function lerp(a,b,x){return a+(b-a)*x;}
  var rs=rng(o.seed||5);
  function mix(c1,c2,x){var a=c1.match(/\w\w/g).map(function(v){return parseInt(v,16);}),b=c2.match(/\w\w/g).map(function(v){return parseInt(v,16);});
    return 'rgb('+a.map(function(v,i){return Math.round(lerp(v,b[i],x));}).join(',')+')';}
  function init(){
    dpr=Math.min(2,window.devicePixelRatio||1);W=cv.clientWidth;H=cv.clientHeight;
    cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
    var nr=o.rows||5,hz=o.horizon==null?0.46:o.horizon;rows=[];
    for(var r=0;r<nr;r++){
      var d=nr>1?r/(nr-1):1, base=H*hz+d*(H*(1-hz))*0.92+H*0.05, gap=lerp(8,16,d), n=Math.ceil(W/gap)+6, st=[];
      for(var i=0;i<n;i++)st.push({x:(i-3)*gap+rs()*gap*0.9,h:lerp(0.13,0.3,d)*H*(o.tall||1)*(0.78+rs()*0.44),ph:rs()*6.28,k:0.7+rs()*0.6});
      rows.push({base:base,d:d,st:st});
    }
    motes=[];for(var m=0;m<(o.motes==null?36:o.motes);m++)motes.push({x:rs()*W,y:rs()*H*0.8,s:0.6+rs()*1.6,p:rs()*6.28});
  }
  function wind(x){
    var g=Math.sin(x*0.0035-t*0.9)*0.5+Math.sin(x*0.009-t*1.7+1.3)*0.22+Math.sin(x*0.0011-t*0.35)*0.4;
    var gust=Math.pow(Math.max(0,Math.sin(t*0.37-x*0.0016)),3)*0.9;
    var dx=(x-mx)/170, m=Math.exp(-dx*dx)*mv;
    return (g+gust+m)*0.5;
  }
  function draw(){
    ctx.clearRect(0,0,W,H);
    var sx=W*0.8,sy=H*0.2;
    ctx.fillStyle='rgba(184,133,31,.12)';ctx.beginPath();ctx.arc(sx,sy,44,0,6.29);ctx.fill();
    ctx.strokeStyle='#b8851f';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(sx,sy,30,0,6.29);ctx.stroke();
    rows.forEach(function(r){
      ctx.fillStyle=mix('e6d9b8','c9b27c',r.d);ctx.fillRect(0,r.base,W,H-r.base+2);
      var stem=mix('c8b27a','8a6a2c',r.d), head=mix('d7b960','a97a22',r.d), lw=lerp(1,2.1,r.d);
      ctx.lineWidth=lw;ctx.strokeStyle=stem;ctx.fillStyle=head;
      r.st.forEach(function(s){
        var w=wind(s.x)*s.k, a=w*1.05+Math.sin(t*2.2+s.ph)*0.03, h=s.h;
        var tx=s.x+Math.sin(a)*h, ty=r.base-Math.cos(a)*h, cx=s.x+Math.sin(a*0.35)*h*0.45, cy=r.base-h*0.62;
        ctx.beginPath();ctx.moveTo(s.x,r.base);ctx.quadraticCurveTo(cx,cy,tx,ty);ctx.stroke();
        var ha=a*1.25, L=lerp(9,17,r.d), c=Math.cos(ha), sn=Math.sin(ha);
        ctx.save();ctx.translate(tx,ty);ctx.rotate(ha);
        ctx.beginPath();ctx.ellipse(0,-L*0.5,lerp(1.6,3.2,r.d),L*0.55,0,0,6.29);ctx.fill();
        ctx.lineWidth=0.7;ctx.beginPath();
        for(var k=-1;k<=1;k+=2){ctx.moveTo(0,-L*0.2);ctx.lineTo(k*lerp(2,4,r.d),-L*1.25);ctx.moveTo(0,-L*0.55);ctx.lineTo(k*lerp(2,4,r.d),-L*1.5);}
        ctx.stroke();ctx.restore();ctx.lineWidth=lw;
      });
    });
    ctx.fillStyle='rgba(184,133,31,.55)';
    motes.forEach(function(m){
      m.x+=(1.0+wind(m.x)*4)*m.s*0.6;m.y+=Math.sin(t*1.3+m.p)*0.25;if(m.x>W+10){m.x=-10;m.y=rs()*H*0.8;}
      ctx.beginPath();ctx.arc(m.x,m.y,m.s*0.9,0,6.29);ctx.fill();
    });
  }
  function frame(ts){
    if(!vis){requestAnimationFrame(frame);return;}
    var dt=Math.min(0.05,(ts-last)/1000||0.016);last=ts;t+=dt;mv*=0.94;draw();requestAnimationFrame(frame);
  }
  init();
  var host=cv.parentElement;
  window.addEventListener('resize',function(){init();draw();});
  host.addEventListener('pointermove',function(e){var r=cv.getBoundingClientRect();mx=e.clientX-r.left;mv=Math.max(-2,Math.min(2,mv+(e.movementX||0)*0.04));});
  host.addEventListener('pointerleave',function(){mx=-9999;});
  if('IntersectionObserver' in window)new IntersectionObserver(function(en){vis=en[0].isIntersecting;}).observe(cv);
  if(reduce){t=2;draw();}else requestAnimationFrame(frame);
}
field($('herofield'),{rows:5,horizon:0.7,seed:11,tall:0.7});
field($('footfield'),{rows:3,horizon:0.25,seed:23,motes:14,tall:1.15});

/* ---------- network diagram ---------- */
(function(){
  var s=$('fig-net');
  var cols=[{x:95,n:11,lab:'22 inputs',sub:'time, prices, link...'},{x:260,n:8,lab:'64 units',sub:'tanh'},{x:430,n:8,lab:'64 units',sub:'tanh (or GRU)'},{x:610,n:11,lab:'24 action scores',sub:'21 prices, accept, walk, wait'}];
  var T0=40,T1=270;
  function ys(n){var a=[];for(var i=0;i<n;i++)a.push(T0+(i+.5)*(T1-T0)/n);return a;}
  var pts=cols.map(function(c){return ys(c.n).map(function(y){return {x:c.x,y:y};});});
  for(var c=0;c<3;c++){
    pts[c].forEach(function(a,i){pts[c+1].forEach(function(b,j){
      if((i*7+j*3)%4!==0)return;
      add(s,'line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:INK,'stroke-width':.8,opacity:.35,'class':'flow',style:'animation-delay:'+(((i+j)%5)*0.25)+'s'});
    });});
  }
  pts.forEach(function(col,ci){col.forEach(function(p,i){
    add(s,'circle',{cx:p.x,cy:p.y,r:ci===3?5:4.6,fill:ci===0?INK:(ci===3?WHEAT:'var(--paper)'),stroke:INK,'stroke-width':1.3});
  });add(s,'text',{x:cols[ci].x,y:T1+22,'text-anchor':'middle','font-size':13,fill:INK,'font-weight':600},cols[ci].lab);
    add(s,'text',{x:cols[ci].x,y:T1+38,'text-anchor':'middle','font-size':11.5,fill:INK3},cols[ci].sub);});
  add(s,'rect',{x:690,y:T0+30,width:56,height:T1-T0-60,fill:'none',stroke:CLAY,'stroke-width':1.4,'stroke-dasharray':'4 3'});
  add(s,'text',{x:718,y:T0+120,'text-anchor':'middle','font-size':12,fill:CLAY,'font-weight':600},'mask');
  add(s,'text',{x:718,y:T0+137,'text-anchor':'middle','font-size':10.5,fill:CLAY},'illegal');
  add(s,'text',{x:718,y:T0+150,'text-anchor':'middle','font-size':10.5,fill:CLAY},'moves out');
  add(s,'line',{x1:746,y1:(T0+T1)/2,x2:800,y2:(T0+T1)/2,stroke:INK,'stroke-width':1.5});
  add(s,'path',{d:'M794 '+((T0+T1)/2-5)+' L802 '+((T0+T1)/2)+' L794 '+((T0+T1)/2+5)+' Z',fill:INK});
  add(s,'text',{x:818,y:(T0+T1)/2-2,'text-anchor':'middle','font-size':13,fill:INK,'font-weight':600},'one');
  add(s,'text',{x:818,y:(T0+T1)/2+14,'text-anchor':'middle','font-size':13,fill:INK,'font-weight':600},'action');
  add(s,'text',{x:430,y:18,'text-anchor':'middle','font-size':11.5,fill:INK3},'two extra heads read the second layer: expected reward and expected fee cost (used for the fairness constraint)');
})();

/* ---------- training replay ---------- */
(function(){
  var OPT=[['x2_gate/SYM','Equal conditions, no rule'],['x2_gate/PERISH','Perishable crop, no rule'],['x3_mechanism/fee_oracle_tau0.4','Perishable crop, adaptive fee'],['x3_mechanism/fee_oracle_tau0.4_eta0.25','Perishable crop, faster fee'],['x3_mechanism/floor_kappa0.55','Perishable crop, price floor 0.55'],['x4_delay/unaware_duty0.6','Rural, link up 60%, delay-unaware'],['x4_delay/aware_duty0.6','Rural, link up 60%, delay-aware']];
  var sel=$('tSel'),sc=$('tScrub'),cur=null,tm=null;
  OPT.forEach(function(o){var e=document.createElement('option');e.value=o[0];e.textContent=o[1];sel.appendChild(e);});
  var W=740,L=44,R=720,T=12,B=210,T2=244,B2=296;
  function draw(){
    var s=$('tsvg');clear(s);var n=cur.length,k=+sc.value;
    function X(i){return L+i/(n-1)*(R-L);} function Y(v){return B-v*(B-T);}
    [0,.25,.5,.75,1].forEach(function(v){add(s,'line',{x1:L,y1:Y(v),x2:R,y2:Y(v),stroke:RULE,'stroke-width':.7});add(s,'text',{x:L-6,y:Y(v)+4,'text-anchor':'end','font-size':11,fill:INK3},v.toFixed(2));});
    var maxl=Math.max(1,Math.max.apply(null,cur.map(function(r){return r[4]||0;})));
    function line(idx,col,w){var d='',pen=false;for(var i=0;i<n;i++){var v=cur[i][idx];if(v==null){pen=false;continue;}d+=(pen?'L':'M')+X(i)+' '+Y(Math.max(0,v))+' ';pen=true;}add(s,'path',{d:d,fill:'none',stroke:col,'stroke-width':w,'stroke-linejoin':'round'});}
    line(3,LEAF,2);line(1,INK,2.2);line(2,CLAY,2.2);
    if(sel.value.indexOf('fee_oracle_tau0.4')>=0)add(s,'line',{x1:L,y1:Y(.4),x2:R,y2:Y(.4),stroke:CLAY,'stroke-dasharray':'3 4','stroke-width':1});
    add(s,'text',{x:L,y:T2-8,'font-size':11,fill:INK3},'fee rate (lambda)');
    add(s,'line',{x1:L,y1:B2,x2:R,y2:B2,stroke:RULE});
    var d2='M'+X(0)+' '+B2+' ';for(var i=0;i<n;i++)d2+='L'+X(i)+' '+(B2-(cur[i][4]||0)/maxl*(B2-T2))+' ';
    add(s,'path',{d:d2+'L'+X(n-1)+' '+B2+' Z',fill:WHEAT,opacity:.35});
    add(s,'path',{d:d2,fill:'none',stroke:WHEAT,'stroke-width':2});
    add(s,'text',{x:L-6,y:T2+6,'text-anchor':'end','font-size':11,fill:INK3},maxl.toFixed(1));
    add(s,'text',{x:L-6,y:B2+4,'text-anchor':'end','font-size':11,fill:INK3},'0');
    add(s,'line',{x1:X(k),y1:T,x2:X(k),y2:B2,stroke:INK,'stroke-width':1.2});
    [[1,INK],[2,CLAY],[3,LEAF]].forEach(function(a){var v=cur[k][a[0]];if(v!=null)add(s,'circle',{cx:X(k),cy:Y(Math.max(0,v)),r:4.5,fill:a[1],stroke:'var(--paper)','stroke-width':1.5});});
    add(s,'circle',{cx:X(k),cy:B2-(cur[k][4]||0)/maxl*(B2-T2),r:4.5,fill:WHEAT,stroke:'var(--paper)','stroke-width':1.5});
    [0,Math.floor(n/2),n-1].forEach(function(i){add(s,'text',{x:X(i),y:B2+16,'text-anchor':i===0?'start':(i===n-1?'end':'middle'),'font-size':11,fill:INK3},'iteration '+cur[i][0]);});
    var r=cur[k];$('tCk').textContent=(k+1)+' of '+n;
    $('tRead').innerHTML='<span>Efficiency <b>'+f2(r[1])+'</b></span><span>Farmer share <b>'+f2(r[2])+'</b></span><span>Deal rate <b>'+f2(r[3])+'</b></span><span>Fee rate <b>'+(r[4]!=null?r[4].toFixed(2):'n/a')+'</b></span>';
  }
  function load(){cur=D.C[sel.value];sc.max=cur.length-1;sc.value=0;stop();draw();}
  function stop(){if(tm){clearInterval(tm);tm=null;}$('tPlay').textContent='Play';}
  $('tPlay').onclick=function(){
    if(tm){stop();return;}
    if(+sc.value>=cur.length-1)sc.value=0;
    $('tPlay').textContent='Pause';
    tm=setInterval(function(){if(+sc.value>=cur.length-1){stop();return;}sc.value=+sc.value+1;draw();},260);
  };
  sc.oninput=function(){stop();draw();};sel.onchange=load;sel.value='x3_mechanism/fee_oracle_tau0.4';load();
  sc.value=cur.length-1;draw();
})();

/* ---------- reveal on scroll ---------- */
(function(){
  var els=[].slice.call(document.querySelectorAll('main section, .finding'));
  els.forEach(function(e){e.classList.add('reveal');});
  if(!('IntersectionObserver' in window)){els.forEach(function(e){e.classList.add('in');});return;}
  var io=new IntersectionObserver(function(en){en.forEach(function(x){if(x.isIntersecting){x.target.classList.add('in');io.unobserve(x.target);}});},{rootMargin:'0px 0px -8% 0px'});
  els.forEach(function(e){io.observe(e);});
})();

/* ---------- side index highlight ---------- */
var links=[].slice.call(document.querySelectorAll('.nav ul a'));
var secs=links.map(function(a){return document.querySelector(a.getAttribute('href'));});
function onScroll(){var y=window.scrollY+120,cur=0;secs.forEach(function(s,i){if(s&&s.offsetTop<=y)cur=i;});links.forEach(function(a,i){a.classList.toggle('on',i===cur);});}
window.addEventListener('scroll',onScroll,{passive:true});onScroll();
})();
