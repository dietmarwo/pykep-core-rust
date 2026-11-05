#!/usr/bin/env python3
"""Compare the SF UDPs in a shared PyGMO NLopt SLSQP convergence study."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pygmo as pg
import pykep as pk
import pykep_rust as rust


def fixture(path):
    with path.open("rb") as stream:
        magic, suite, rows, width = stream.readline().decode().split()
        if magic!="PYKEP_BENCH_V1" or not suite.startswith("sf"):
            raise ValueError("expected a Sims-Flanagan fixture")
        rows, width = int(rows), int(width)
        if not 0<rows*width<16_777_216 or width<22 or (width-19)%3:
            raise ValueError("invalid SF fixture dimensions")
        values = np.fromfile(stream, dtype="<f8", count=rows*width)
        if stream.read(1) or not np.isfinite(values).all():
            raise ValueError("malformed fixture")
    return suite, values.reshape(rows,width)


class SfUdp:
    def __init__(self, row, backend, derivative):
        self.row = row
        self.backend = backend
        self.derivative = derivative
        self.n = len(row)-19
        self.fitness_calls = 0
        self.gradient_calls = 0
        self.leg_constructions = 0

    def leg(self, x):
        self.leg_constructions += 1
        r = self.row
        if self.backend=="rust":
            return rust.SimsFlanaganLeg(r[:6],r[6],x[:self.n].reshape(-1,3),r[7:13],
                x[-1],x[-2],r[15],r[16],r[17],r[18])
        return pk.leg.sims_flanagan(r[:6].reshape(2,3),r[6],x[:self.n],r[7:13].reshape(2,3),
                                   x[-1],x[-2],r[15],r[16],r[17],r[18])

    def fitness(self, x):
        self.fitness_calls += 1
        leg = self.leg(x)
        if self.backend=="rust":
            return np.r_[-x[-1],leg.mismatch_constraints(),leg.throttle_constraints()]
        return np.r_[-x[-1],leg.compute_mismatch_constraints(),leg.compute_throttle_constraints()]

    def gradient(self, x):
        self.gradient_calls += 1
        if self.derivative=="numerical":
            return pg.estimate_gradient(self.fitness,x)
        leg=self.leg(x)
        if self.backend=="rust":
            _,end,control=leg.mismatch_jacobian()
        else:
            _,end,control=leg.compute_mc_grad()
        gradient=np.zeros((8+self.n//3,len(x)))
        gradient[0,-1]=-1.
        gradient[1:8,:-1]=np.reshape(control,(7,self.n+1))
        gradient[1:8,-1]=np.reshape(end,(7,7))[:,6]
        for j in range(self.n//3):
            gradient[8+j,3*j:3*j+3]=2*x[3*j:3*j+3]
        return gradient.ravel()

    def get_bounds(self):
        return ([-1.]*self.n+[np.pi/12,.5],[1.]*self.n+[2*np.pi,1.])

    def get_nec(self):
        return 7

    def get_nic(self):
        return self.n//3


def converge(row, backend, derivative, limit):
    udp=SfUdp(row,backend,derivative)
    x=np.r_[row[19:],row[14],row[13]]
    pop=pg.population(pg.problem(udp),0)
    pop.problem.c_tol=1e-8
    pop.push_back(x)
    solver=pg.nlopt("slsqp")
    solver.xtol_abs=0.
    solver.xtol_rel=0.
    solver.ftol_abs=0.
    solver.ftol_rel=0.
    solver.maxeval=limit
    alg=pg.algorithm(solver)
    start=perf_counter()
    try:
        solution=alg.evolve(pop)
        elapsed=perf_counter()-start
        actual=solution.problem.extract(SfUdp)
        fitness=udp.fitness(solution.champion_x)
        feasible=bool(np.max(np.abs(fitness[1:8]))<=1e-8 and np.max(fitness[8:])<=1e-8)
        return {"status":"completed","elapsed_seconds":elapsed,"feasible":feasible,
                "x":solution.champion_x.tolist(),"raw_fitness":fitness.tolist(),
                "solver_result":int(alg.extract(pg.nlopt).get_last_opt_result()),
                "fitness_calls":actual.fitness_calls,"gradient_calls":actual.gradient_calls,
                "leg_constructions":actual.leg_constructions}
    except Exception as exc:
        return {"status":"failed","elapsed_seconds":perf_counter()-start,"feasible":False,"error":str(exc)}


def timing(row, calls):
    output=[]
    for backend in ("rust","upstream"):
        for derivative in ("analytic","numerical"):
            udp=SfUdp(row,backend,derivative)
            x=np.r_[row[19:],row[14],row[13]]
            problem=pg.problem(udp)
            for label, call in (("direct",udp.gradient),("pygmo",problem.gradient)):
                call(x)
                elapsed=[]
                for _ in range(7):
                    start=perf_counter()
                    for _ in range(calls):
                        call(x)
                    elapsed.append((perf_counter()-start)*1e6/calls)
                output.append({"backend":backend,"derivative":derivative,"boundary":label,
                               "samples_us":elapsed})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--count",type=int,default=100)
    parser.add_argument("--maxeval",type=int,default=1000)
    parser.add_argument("--wrapper-timing",action="store_true")
    args=parser.parse_args()
    suite, rows=fixture(args.fixture)
    report={"suite":suite,"pykep":pk.__version__,"pygmo":pg.__version__,
            "optimizer":"PyGMO NLopt SLSQP","maxeval":args.maxeval,"c_tol":1e-8,"runs":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.wrapper_timing:
        report["gradient_wrapper_timing"]=timing(rows[0],32)
    else:
        for i,row in enumerate(rows[:args.count]):
            for backend in ("rust","upstream"):
                for derivative in ("analytic","numerical"):
                    report["runs"].append({"case":i,"backend":backend,"derivative":derivative,
                                           **converge(row,backend,derivative,args.maxeval)})
            if i%10==0:
                print(suite,"cases",i+1,flush=True)
            write(args.output,report)
    write(args.output,report)


def write(path,value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


if __name__=="__main__":
    main()
