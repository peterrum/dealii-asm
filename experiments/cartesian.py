import json
import os
from argparse import ArgumentParser

def run_instance(counter, d, l, k, solver, preconditioner, sequence, s):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["name"]          = solver.lower() + "-" + preconditioner
    datastore["mesh"]["name"]  = "hypercube"
    datastore["dim"]           = d
    datastore["n refinements"] = l
    datastore["degree"]        = k

    # ... solver
    datastore["solver"]["type"] = solver

    # ... multigrid
    datastore["preconditioner"]["mg p sequence"]      = sequence
    datastore["preconditioner"]["mg smoother"]["degree"] = s

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

        if props[3] == "f":
          datastore["preconditioner"]["mg smoother"]["preconditioner"]["sub mesh approximation"] = d
        else:
          datastore["preconditioner"]["mg smoother"]["preconditioner"]["sub mesh approximation"] = 1

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")

    parser.add_argument('d', type=int)
    parser.add_argument('l', type=int)
    parser.add_argument('k', type=int)
    
    arguments = parser.parse_args()
    return arguments

def main():
    options = parseArguments()

    d = options.d
    l = options.l
    k = options.k
    
    counter = 0

    preconditioners = ["diagonal"]

    for a in ["none", "pre", "post", "symm"]:
        for o in range(1, min(k, 3)+1):
            preconditioners.append("fdm-%s-%d-f" % (a, o));

        for o in range(2, min(k, 3)+1):
            preconditioners.append("fdm-%s-%d-r" % (a, o));

    for solver in ["CG", "GMRES"]:
        preconditioners_to_be_used = preconditioners

        if solver == "CG":
            preconditioners_to_be_used = [i for i in preconditioners_to_be_used if ("diagonal" == i) or  ("symm" in i) or ("none" in i) ]

        for preconditioner in preconditioners_to_be_used:
            for sequence in ["bisect", "go to one", "decrease by one"]:
                for s in range(1, 6):
                    run_instance(counter, d, l, k, solver, preconditioner, sequence, s)
                    counter = counter + 1;


if __name__== "__main__":
  main()
  