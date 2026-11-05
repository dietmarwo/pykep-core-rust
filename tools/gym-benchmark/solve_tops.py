#!/usr/bin/env python3
"""Regenerate a feasible TOPS replay vector with a bounded SciPy study."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from pykep_rust.trajopt import gym


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem")
    parser.add_argument("--nseg",type=int,default=20)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--evaluations",type=int,default=300)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--initial",type=Path)
    args=parser.parse_args()
    options={"nseg":args.nseg}
    p=gym.Problem(args.problem,**options)
    spec=p.metadata()
    low,high=np.array(p.get_bounds())
    free=low<high
    rng=np.random.default_rng(args.seed)
    if args.initial:
        x=np.array(json.loads(args.initial.read_text())["x"])
    else:
        x=low+(.15+.7*rng.random(len(low)))*(high-low)
        for i,name in enumerate(spec["decisions"]):
            if name.startswith("direction_") and name.endswith("_x"):
                d=rng.normal(size=3)
                x[i:i+3]=d/np.linalg.norm(d)
            if name.startswith("alpha_"):
                x[i]=.28+.18*rng.normal()
    assert x.shape==low.shape
    nec=p.get_nec()
    nobj=p.get_nobj()
    calls={"fitness":0,"gradient":0}
    previous=None
    cached=None

    def unpack(z):
        y=low.copy()
        y[free] += z*(high[free]-low[free])
        return y

    def fitness(z):
        nonlocal previous,cached
        if previous is None or not np.array_equal(z,previous):
            previous=z.copy()
            calls["fitness"] += 1
            cached=np.asarray(p.fitness(unpack(z)))[nobj:]
        return cached

    def residual(z):
        g=fitness(z)
        return np.r_[g[:nec],np.maximum(g[nec:],0.)]

    def jacobian(z):
        g=fitness(z)
        calls["gradient"] += 1
        j=np.asarray(p.gradient(unpack(z))).reshape(-1,len(low))[nobj:]
        j[nec:] *= (g[nec:]>0.)[:,None]
        return j[:,free]*(high[free]-low[free])

    start=perf_counter()
    start_point=np.clip((x[free]-low[free])/(high[free]-low[free]),1e-12,1-1e-12)
    solution=least_squares(residual,start_point,jac=jacobian,bounds=(0.,1.),
                          max_nfev=args.evaluations,ftol=1e-12,gtol=1e-13,xtol=1e-12)
    x=unpack(solution.x)
    f=np.asarray(p.fitness(x))
    forward=p.forward_residual(x)
    feasible=bool(np.max(np.abs(f[nobj:nobj+nec]),initial=0.)<=1e-8
                  and np.max(f[nobj+nec:],initial=0.)<=1e-8 and max(map(abs,forward))<=1e-6)
    result={"id":args.problem,"options":options,"x":x.tolist(),"fitness":f.tolist(),
            "forward_residual":forward,"feasible":feasible,"seed":args.seed,
            "calls":calls,"elapsed_seconds":perf_counter()-start,"message":solution.message,
            "solver":"SciPy least_squares TRF with analytic Rust Jacobian",
            "requested_evaluations":args.evaluations}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(args.problem, feasible, "max forward residual", max(map(abs,forward)), calls, flush=True)
    return 0 if feasible else 1


if __name__=="__main__":
    raise SystemExit(main())
