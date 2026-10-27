"""Causal confirmation of consecutive observed positive alert scores."""
from __future__ import annotations

import numpy as np


def apply_confirmation(raw_score: np.ndarray,elapsed: np.ndarray,gap_before: np.ndarray,
                       confirmation_seconds: float) -> dict:
    raw=np.asarray(raw_score,dtype=float)
    elapsed=np.asarray(elapsed,dtype=float)
    gaps=np.asarray(gap_before,dtype=bool)
    if raw.ndim!=1 or elapsed.shape!=raw.shape or gaps.shape!=raw.shape:
        raise ValueError('Policy scores, elapsed time and gap flags must be matching vectors.')
    duration=float(confirmation_seconds)
    if not np.isfinite(duration) or duration<0:
        raise ValueError('Confirmation duration must be finite and nonnegative.')
    if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed)<=0):
        raise ValueError('Policy elapsed timestamps must be finite and strictly increasing.')
    decision=raw.copy();pending=np.zeros(len(raw),bool);confirmed=np.zeros(len(raw),bool)
    starts=np.full(len(raw),-1,dtype=int)
    start=-1;active=False
    for index,value in enumerate(raw):
        if gaps[index]:
            start=-1;active=False
        if not np.isfinite(value) or value<=1.:
            start=-1;active=False
            continue
        if start<0:
            start=index
        starts[index]=start
        if active or elapsed[index]-elapsed[start]>=duration:
            active=True;confirmed[index]=True
        else:
            pending[index]=True
            decision[index]=min(value,1.)
    return {'raw_score':raw.copy(),'decision_score':decision,'pending':pending,'confirmed':confirmed,
            'raw_episode_start':starts,'confirmation_seconds':duration}
