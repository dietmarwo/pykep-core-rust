#!/usr/bin/env python3
"""Bounded reproductions of the two upstream benchmark notebooks."""

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import pykep as pk
import pykep_rust as rust

PIN="c01e19debad5ea22b7079189f786f473046a79d5"


def notebook(source, name):
    path=source/"benchmark"/"notebooks"/(name+".ipynb")
    return path,json.loads(path.read_text())["cells"]


def execute(path,cells,selected,env):
    for index in selected:
        cell=cells[index]
        assert cell["cell_type"]=="code"
        source="".join(cell["source"])
        exec(compile(source,f"{path}#cell-{index}","exec"),env)


def compare(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(np.abs(a-b)/np.maximum(1.,np.abs(b))))


def symbolic(source):
    path,cells=notebook(source,"sf_gradient")
    env={}
    execute(path,cells,[1,2,4,6,7,9,12,14],env)
    leg=rust.SimsFlanaganLeg(
        np.r_[env["rs"],env["vs"]],env["ms"],env["throttles"].reshape(-1,3),
        np.r_[env["rf"],env["vf"]],env["mf"],env["tof"],env["max_thrust"],env["veff"],env["mu"],.6,
    )
    # The pinned cell assigns the old Isp property. Its symbolic input has
    # already converted Isp to exhaust velocity, which the current leg takes.
    upstream=pk.leg.sims_flanagan([env["rs"],env["vs"]],env["ms"],env["throttles"],
        [env["rf"],env["vf"]],env["mf"],env["tof"],env["max_thrust"],env["veff"],env["mu"],.6)
    own=leg.mismatch_jacobian()
    oracle=upstream.compute_mc_grad()
    symbolic=[env[k] for k in ("mcg_xs_hy","mcg_xf_hy","mcg_th_tof_hy")]
    return {"notebook_adapter":"cell 16 uses current veff constructor",
            "rust_vs_upstream_gradient":[compare(a,np.reshape(b,np.shape(a))) for a,b in zip(own,oracle)],
            "rust_vs_symbolic_gradient":[compare(a,np.reshape(b,np.shape(a))) for a,b in zip(own,symbolic)],
            "rust_vs_upstream_mismatch":compare(leg.mismatch_constraints(),upstream.compute_mismatch_constraints())}


def sampled(call,repeats=7):
    call()
    samples=[]
    for _ in range(repeats):
        start=perf_counter()
        value=call()
        samples.append(perf_counter()-start)
    return value,samples


def interface(source,count):
    path,cells=notebook(source,"benchmarkudplaiface")
    env={}
    execute(path,cells,[0,1],env)
    epochs=np.asarray(env["mjd2000s"][:count])
    original={}
    for name,label,setup,body,scale in [
        ("TLE","TLE",[2,3,4,5,6],[],["udplaeph","plaeph","plaephv","sgp4p","sgp4pv"]),
        ("SPICE","SPICE",[8],[9,10,11,12,13],["udplaeph","plaeph","plaephv","spicep","spicepv"]),
    ]:
        arm={**env,"mjd2000s":epochs}
        try:
            if name=="SPICE":
                text="".join(cells[8]["source"]).replace("../../pykep/data/de440s.bsp",
                                                           str(source/"pykep/data/de440s.bsp"))
                exec(compile(text,f"{path}#cell-8","exec"),arm)
            else:
                execute(path,cells,setup,arm)
            failures={}
            for index in body:
                try:
                    execute(path,cells,[index],arm)
                except Exception as exc:
                    failures[str(index)] = str(exc)
            original[label]={"status":"reference-only","seconds":{key:arm[key] for key in scale if key in arm},
                             "failed_cells":failures}
        except Exception as exc:
            original[label]={"status":"failed","error":str(exc)}
    providers=[]
    # Measure the equivalent dispatch boundaries for supported Rust ephemerides.
    earth = pk.planet(pk.udpla.jpl_lp("earth"))
    state = earth.eph(0.)
    providers_to_measure = [
        ("Keplerian",rust.Planet.keplerian_from_state(0.,np.ravel(state),pk.MU_SUN),
         pk.planet(pk.udpla.keplerian(pk.epoch(0.),state,pk.MU_SUN))),
        ("JPL",rust.Planet.jpl_low_precision("earth"),earth),
        ("VSOP",rust.Planet.vsop2013("earth_moon"),pk.planet(pk.udpla.vsop2013("earth_moon"))),
    ]
    for name,provider,reference in providers_to_measure:
        values,scalar=sampled(lambda: [provider.state(float(t)) for t in epochs])
        batches,batch=sampled(lambda:provider.states(epochs,workers=1))
        ref,ref_scalar=sampled(lambda:[np.ravel(reference.eph(float(t))) for t in epochs])
        ref_batch,ref_batch_time=sampled(lambda:np.reshape(reference.eph_v(epochs),(-1,6)))
        providers.append({"name":name,"count":len(epochs),"rust_scalar_seconds":scalar,
            "rust_batch_seconds":batch,"upstream_scalar_seconds":ref_scalar,
            "upstream_batch_seconds":ref_batch_time,"rust_scalar_batch_error":compare(values,batches),
            "upstream_scalar_batch_error":compare(ref,ref_batch),"upstream_scaled_difference":compare(values,ref)})
    return {"upstream_reference_only":original,"rust_provider_analogue":providers}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream",type=Path)
    parser.add_argument("study",choices=("symbolic","interface"))
    parser.add_argument("--count",type=int,default=20000)
    args=parser.parse_args()
    source=args.upstream.resolve()
    if subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()!=PIN:
        parser.error("expected the pinned source checkout")
    print(json.dumps(symbolic(source) if args.study=="symbolic" else interface(source,args.count),
                     allow_nan=False,indent=2))


if __name__=="__main__":
    main()
