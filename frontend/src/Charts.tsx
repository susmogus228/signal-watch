import {useEffect,useRef} from 'react';
import * as echarts from 'echarts/core';
import {LineChart,CustomChart} from 'echarts/charts';
import {GridComponent,TooltipComponent,DataZoomComponent,MarkAreaComponent,MarkLineComponent,AriaComponent} from 'echarts/components';
import {CanvasRenderer} from 'echarts/renderers';
import type {EChartsOption} from 'echarts';
import {time,type View,type Event,type Method} from './api';
echarts.use([LineChart,CustomChart,GridComponent,TooltipComponent,DataZoomComponent,MarkAreaComponent,MarkLineComponent,AriaComponent,CanvasRenderer]);

const colors={learned:'#edb347',robust:'#aab4c3',persistence:'#7f9b97'};
const label={color:'#aeb5bf',fontSize:11,fontFamily:'ui-monospace, SFMono-Regular, Consolas, monospace'};
const axes={type:'value' as const,axisLabel:{...label,formatter:(v:number)=>time(v)},axisLine:{show:true,lineStyle:{color:'#58606b'}},splitLine:{lineStyle:{color:'#2a3036',type:'dashed' as const}},min:0};

export function Chart({option,height,label:summary}:{option:EChartsOption;height:number;label:string}){
 const ref=useRef<HTMLDivElement>(null);
 useEffect(()=>{if(!ref.current)return;const chart=echarts.init(ref.current,undefined,{renderer:'canvas'});chart.group='signalwatch-recording';echarts.connect('signalwatch-recording');chart.setOption({...option,animation:false,backgroundColor:'transparent',textStyle:{fontFamily:'system-ui, sans-serif'},aria:{enabled:true,description:summary}});const resize=new ResizeObserver(()=>chart.resize());resize.observe(ref.current);return()=>{resize.disconnect();chart.dispose();};},[option,summary]);
 return <div ref={ref} className="chart" role="img" aria-label={summary} style={{height}}/>;
}
function common(max:number):EChartsOption{return {dataZoom:[{type:'inside',xAxisIndex:0,filterMode:'none'}],tooltip:{trigger:'axis',backgroundColor:'#1a1a1a',borderColor:'#454b51',textStyle:{color:'#eee'}},grid:{left:50,right:22,top:15,bottom:44},xAxis:{...axes,max:Math.max(1,max),name:'Elapsed time (mm:ss)',nameLocation:'middle',nameGap:29,nameTextStyle:{color:'#aeb5bf',fontSize:11}},yAxis:{type:'value',scale:true,axisLabel:label,splitLine:{lineStyle:{color:'#2a3036',type:'dashed'}}}};}
function highlighted(view:View,event:Event|undefined){const areas:unknown[]=[];if(event)areas.push([{xAxis:event.start_seconds,itemStyle:{color:'#edb34715'}},{xAxis:event.end_seconds}]);view.gaps.forEach(g=>{const x=view.elapsed[g.index];if(x!==undefined)areas.push([{xAxis:view.elapsed[Math.max(0,g.index-1)],itemStyle:{color:'#c26b7330'}},{xAxis:x}]);});return {silent:true,data:areas} as never;}
function points(view:View,values:(number|null)[]){const gap=new Set(view.gaps.map(g=>g.index));return view.elapsed.flatMap((x,i)=>gap.has(i)?[[x,null],[x,values[i]]]:[[x,values[i]]]);}
export function SignalChart({view,sensor,event}:{view:View;sensor:string;event?:Event}){
 const s=view.signals.find(s=>s.key===sensor)??view.signals[0];if(!s)return <div className="chart-empty">Awaiting observations. Step or play the recording.</div>;
 const option:EChartsOption={...common(view.elapsed.at(-1)??0),series:[{type:'line',name:`${s.label} (${s.unit})`,data:points(view,s.values),showSymbol:false,connectNulls:false,lineStyle:{color:'#eee9df',width:1.5},itemStyle:{color:'#eee9df'},markArea:highlighted(view,event),...(event?.first_alert_seconds!=null?{markLine:{symbol:'none',lineStyle:{color:'#edb347',type:'dashed'},label:{color:'#edb347',formatter:`First alert ${time(event.first_alert_seconds)}`},data:[{xAxis:event.first_alert_seconds}]}}:{})}]};
 return <Chart option={option} height={194} label={`${s.label}, ${s.unit}. ${view.cursor} recorded samples. Highlighted episode ${event?`${time(event.start_seconds)} to ${time(event.end_seconds)}`:'none'}. Gaps break the line.`}/>;
}
export function ScoreChart({view,event}:{view:View;event?:Event}){
 const option:EChartsOption={...common(view.elapsed.at(-1)??0),yAxis:{type:'value',min:0,axisLabel:label,splitLine:{lineStyle:{color:'#2a3036',type:'dashed'}}},series:(Object.keys(colors) as Method[]).map((key,i)=>({type:'line',name:key==='learned'?(view.alarm_policy?'Confirmed decision':'Learned detector'):key==='robust'?'Rolling median / MAD':'Persistence',data:points(view,view.scores[key]??[]),showSymbol:false,connectNulls:false,lineStyle:{color:colors[key],width:i===0?2:1.4,type:i===1?'dashed':'solid'},itemStyle:{color:colors[key]},...(i===0?{markArea:highlighted(view,event),markLine:{symbol:'none',lineStyle:{color:'#cd9560',type:'dashed'},label:{show:false},data:[{yAxis:1}]}}:{})}))};
 if(view.raw_scores?.learned&&Array.isArray(option.series))option.series=[{type:'line',name:'Raw ML score',data:points(view,view.raw_scores.learned),showSymbol:false,connectNulls:false,lineStyle:{color:'#858d9c',width:1.2,type:'dotted'},itemStyle:{color:'#858d9c'}},...option.series];
 return <Chart option={option} height={159} label={`${view.alarm_policy?'Raw ML is dotted; the confirmed decision caps pending scores at 1. ':''}Relative alert levels. ${Object.entries(view.score_normalization??{}).map(([key,formula])=>`${key}: ${formula}`).join("; ")}. A score above 1 triggers an alert. Scores are not probabilities. Warm-up after each gap: learned ${view.warmup_by_method?.learned??view.warmup_samples} samples; baselines ${view.warmup_by_method?.robust??view.warmup_samples} samples.`}/>;
}
export function LanesChart({view}:{view:View}){
 const data:number[][]=[];const items=[view.mode==='audit'?view.annotations:[],...(view.alarm_policy?[view.raw_episodes??[]]:[]),view.episodes.learned??[],view.episodes.robust??[],view.episodes.persistence??[]];items.forEach((episodes,y)=>episodes.forEach(e=>data.push([e.start_seconds,Math.max(e.start_seconds+.3,e.end_seconds),y])));
 const option:EChartsOption={...common(view.elapsed.at(-1)??0),grid:{left:50,right:22,top:7,bottom:41},yAxis:{type:'category',data:view.alarm_policy?['Labels','Raw ML','Confirm','MAD','Persist.']:['Labels','Learned','MAD','Persist.'],inverse:true,axisLabel:{color:'#b7bdc8',fontSize:11},axisLine:{show:false},axisTick:{show:false},splitLine:{show:true,lineStyle:{color:'#292f35'}}},series:[{type:'custom',data,renderItem:(params,api)=>{const start=api.coord([api.value(0),api.value(2)]),end=api.coord([api.value(1),api.value(2)]);return {type:'rect',shape:{x:start[0],y:start[1]-5,width:Math.max(2,end[0]-start[0]),height:10},style:{fill:(view.alarm_policy?['#93753f','#858d9c','#edb347','#aab4c3','#7f9b97']:['#93753f','#edb347','#aab4c3','#7f9b97'])[Number(api.value(2))],opacity:.9}};}}]};
 return <Chart option={option} height={view.alarm_policy?166:138} label={`Aligned event timeline. ${view.alarm_policy?'Raw ML and confirmed alarms occupy separate lanes. ':''} ${view.mode==='replay'?'Known labels are hidden during replay.':view.recording.labelled?'Known labels are visible for retrospective audit.':'Reference labels are unavailable; correctness is unknown.'} All detector lanes share the elapsed time axis.`}/>;
}
