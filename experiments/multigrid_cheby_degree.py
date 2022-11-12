import json
import os
from argparse import ArgumentParser

def run_instance(counter, d, l, k, sweep, solver, preconditioner, sequence, s, eps):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["name"]            = solver.lower() + "-" + preconditioner
    datastore["mesh"]["name"]    = "anisotropy"
    datastore["mesh"]["stratch"] = eps
    datastore["dim"]             = d
    datastore["n refinements"]   = l
    datastore["degree"]          = k

    # ... solver
    datastore["solver"]["type"] = solver

    # ... multigrid
    datastore["preconditioner"]["mg p sequence"]      = sequence
    datastore["preconditioner"]["mg smoother"]["degree"] = s
    datastore["preconditioner"]["mg smoother"]["type"] = sweep

    # ... preconditioner of smoother
    if preconditioner == "diagonal":
        datastore["preconditioner"]["mg smoother"]["preconditioner"]["type"] = "Diagonal"
    else:
        props = preconditioner.split("-")

        if props[0] != "fdm":
            raise Exception("Not implemented!")

        datastore["preconditioner"]["mg smoother"]["preconditioner"]["type"] = "FDM"
        datastore["preconditioner"]["mg smoother"]["preconditioner"]["weighting type"] = props[1]
        datastore["preconditioner"]["mg smoother"]["preconditioner"]["n overlap"] = props[2]

        if props[1] == "post":
            datastore["preconditioner"]["mg smoother"]["degree"] = 2 * s
            datastore["preconditioner"]["one-sided v-cycle"] = True

        if props[3] == "f":
          datastore["preconditioner"]["mg smoother"]["preconditioner"]["sub mesh approximation"] = d
        else:
          datastore["preconditioner"]["mg smoother"]["preconditioner"]["sub mesh approximation"] = 1

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():

    d = 3
    l = 6
    k = 4
    
    counter = 0

    preconditioners = []

    for a in ["symm"]:
        for o in range(1, 2):
            preconditioners.append("fdm-%s-%d-f" % (a, o));

    for eps in [50.0]:
        for solver in ["CG", "GMRES"]:
            preconditioners_to_be_used = preconditioners

            if solver == "CG":
                preconditioners_to_be_used = [i for i in preconditioners if ("diagonal" == i) or  ("symm" in i)]
            elif solver == "GMRES":
                preconditioners_to_be_used = [i for i in preconditioners if ("post" in i)]

            for sweep in ["Chebyshev"]:
                for preconditioner in preconditioners_to_be_used:
                    for s in range(1, 6):
                        for sequence in ["bisect", "go to one", "decrease by one"]:
                            run_instance(counter, d, l, k, sweep, solver, preconditioner, sequence, s, eps)
                            counter = counter + 1;


if __name__== "__main__":
  main()
  