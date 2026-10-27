import {display} from './api';

export function ModelSummary({model}:{model:Record<string,unknown>}){
 const fields:[string,unknown][]=[
  ['Algorithm',model.algorithm],['Unique training examples',model.training_rows],
  ['Minimum history',`${display(model.window)} samples`],['Input features',model.feature_count],
  ['Training scope',model.training_scope],['Dataset snapshot',model.dataset_snapshot],
  ['Model SHA-256',model.model_sha256],
 ];
 const calibrated=Number(model.study_version)===7;const hybrid=Number(model.study_version)>=5;const policy=model.alarm_policy as {confirmation_seconds?:number}|undefined;
 return <>{calibrated&&<p className="note">Three classifier/calibrator pairs average their outputs. Each member uses separate fitting and calibration recordings; nested grouped evaluation selects thresholds. The normal-behavior predictor keeps its original configuration. Scores do not establish physical-failure probabilities.</p>}{policy&&<p className="note">Both ML branches use the V5 training configuration and thresholds. Alarms require consecutive positive observations spanning {policy.confirmation_seconds} seconds. Raw ML scores remain available; confirmation adds a measured delay.</p>}<dl className="model-summary">{fields.map(([label,value])=><div key={label}><dt>{label}</dt><dd className={label.includes('SHA')||label.includes('snapshot')?'mono hash':''}>{String(value??'Unavailable')}</dd></div>)}</dl><p className="small muted">{hybrid?'Two trained branches combine known-state classification with normal-behavior prediction errors. Fusion thresholds use grouped out-of-fold development scores; the displayed score is a relative alert level.':'The learned model uses labelled development recordings. Its threshold is selected with grouped out-of-fold scores; classifier scores are uncalibrated.'} Baselines retain normal-block calibration. A hit requires a new alert onset inside the annotation.</p></>;
}
