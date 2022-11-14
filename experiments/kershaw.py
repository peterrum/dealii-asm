import json
import os
from argparse import ArgumentParser

def run_instance(counter, d, l, k, solver, preconditioner, sequence, s, eps, cheby_kind, cycle_type):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["name"]          = solver.lower() + "-" + preconditioner + "-" + cheby_kind.replace(" ", "_") + "-" + cycle_type.replace(" ", "_")
    datastore["mesh"]["name"]  = "kershaw"
    datastore["mesh"]["eps"]   = eps
    datastore["dim"]           = d
    datastore["n refinements"] = l
    datastore["degree"]        = k

    if eps != 1.0:
        datastore["operator mapping type"] = "quadratic geometry"
        datastore["mapping degree"]        = 2

    # ... solver
    datastore["solver"]["type"] = solver

    # ... multigrid
    datastore["preconditioner"]["mg p sequence"]                  = sequence
    datastore["preconditioner"]["mg smoother"]["degree"]          = s
    datastore["preconditioner"]["mg smoother"]["polynomial type"] = cheby_kind

    # ... preconditioner of smoother
    if preconditioner == "diagonal":
        datastore["preconditioner"]["mg smoother"]["preconditioner"]["type"] = "Diagonal"
    else:
        props = preconditioner.split("_")

        if props[0] != "fdm":
            raise Exception("Not implemented!")

        datastore["preconditioner"]["mg smoother"]["preconditioner"]["type"] = "FDM"
        datastore["preconditioner"]["mg smoother"]["preconditioner"]["weighting type"] = props[1]

        if props[2] == "v":
            datastore["preconditioner"]["mg smoother"]["preconditioner"]["element centric"] = False
        else:
            datastore["preconditioner"]["mg smoother"]["preconditioner"]["n overlap"] = props[2]

        if cycle_type == "one sided":
            datastore["preconditioner"]["mg smoother"]["degree"] = 2 * s
            datastore["preconditioner"]["one-sided v-cycle"] = True

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

    for a in ["post", "symm"]:
        for o in range(1, 3):
            preconditioners.append("fdm_%s_%d_f" %  (a, o));
        preconditioners.append("fdm_%s_v_f" % (a));

        #for o in range(1, 2):
        #    preconditioners.append("fdm-%s-%d-r" %  (a, o));

    #for eps in [1.0, 0.99, 0.9, 0.7, 0.5, 0.3]:
    for eps in [1.0, 0.3]:
        for solver in ["CG", "GMRES"]:
            preconditioners_to_be_used = []

            if solver == "CG":
                preconditioners_to_be_used = [i for i in preconditioners if ("diagonal" == i) or  ("symm" in i)]
            elif solver == "GMRES":
                preconditioners_to_be_used = [i for i in preconditioners if ("post" in i)]

            for preconditioner in preconditioners_to_be_used:
                for cheby_kind in ["1st kind", "4th kind"]:
                
                    if solver == "CG":
                        cycle_types = ["two sided"]
                    elif solver == "GMRES":
                        cycle_types = ["two sided", "one sided"]

                    for cycle_type in cycle_types:
                        for sequence in ["bisect", "go to one", "decrease by one"]:
                            for s in range(1, 6):
                                run_instance(counter, d, l, k, solver, preconditioner, sequence, s, eps, cheby_kind, cycle_type)
                                counter = counter + 1;


if __name__== "__main__":
  main()
  