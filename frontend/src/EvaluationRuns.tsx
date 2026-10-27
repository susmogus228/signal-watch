import {useState} from 'react';
import {ArrowRight} from '@phosphor-icons/react';
import {display,time,type Event} from './api';

type Run = {id:string;group:string;metrics?:Record<string,Record<string,unknown>>;events?:Event[]};
export function EvaluationRuns({value,onOpen}:{value:unknown;onOpen:(id:string,eventId?:string)=>Promise<void>}){
 const [limit,setLimit]=useState(6),[busy,setBusy]=useState(false),[error,setError]=useState('');
 const report=value as {per_recording?:Run[]};
 const runs=Array.isArray(report?.per_recording)?report.per_recording:[];
 if(!runs.length)return null;
 const open=async(id:string,eventId?:string)=>{setBusy(true);setError('');try{await onOpen(id,eventId);}catch(e){setError(e instanceof Error?e.message:'Could not open this recording.');}finally{setBusy(false);}};
 return <section className="content-panel"><h2>Inspect each outcome</h2><p className="muted">Open a held-out recording or an individual learned-detector outcome directly from the measured report.</p>{error&&<p role="alert" className="red-text">{error}</p>}<div className="run-evidence-list">{runs.slice(0,limit).map(run=><details key={run.id} className="run-evidence"><summary><strong>{run.id}</strong><span>{display(run.metrics?.learned?.detected_events)} / {display(run.metrics?.learned?.total_events)} events · {display(run.metrics?.learned?.false_episodes)} false onsets</span></summary><button disabled={busy} onClick={()=>void open(run.id)}>Open recording <ArrowRight size={15}/></button><ul>{(run.events??[]).map(event=><li key={event.id}><div><strong>{event.title}</strong><span className="muted">{event.status.replaceAll('_',' ')} · {time(event.start_seconds)}–{time(event.end_seconds)}</span></div><button disabled={busy} aria-label={`Inspect ${event.id}`} onClick={()=>void open(run.id,event.id)}>Inspect event <ArrowRight size={14}/></button></li>)}</ul>{!run.events?.length&&<p className="muted small">No learned alert or annotated event in this recording.</p>}</details>)}</div>{runs.length>limit&&<button onClick={()=>setLimit(runs.length)}>Show all {runs.length} recordings</button>}</section>;
}
