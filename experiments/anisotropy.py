import json
import os
import sys

def run_instance(counter, d, l, k, solver, preconditioner, sequence, s, eps, cheby_kind, cycle_type):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["name"]            = solver.lower() + "-" + preconditioner + "-" + cheby_kind.replace(" ", "_") + "-" + cycle_type.replace(" ", "_")
    datastore["mesh"]["name"]    = "anisotropy"
    datastore["mesh"]["stratch"] = eps
    datastore["dim"]             = d
    datastore["n refinements"]   = l
    datastore["degree"]          = k

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

            datastore["preconditioner"]["mg intermediate smoother"] = {}
            datastore["preconditioner"]["mg intermediate smoother"]["preconditioner"] = {}

            datastore["preconditioner"]["mg intermediate smoother"]["preconditioner"]["type"] = "Diagonal"
            datastore["preconditioner"]["mg intermediate smoother"]["type"]                   = "Chebyshev"

            if cycle_type == "one sided":
                datastore["preconditioner"]["mg intermediate smoother"]["degree"] = 2 * (s + 2)
            else:
                datastore["preconditioner"]["mg intermediate smoother"]["degree"] = s + 2

            datastore["preconditioner"]["mg intermediate smoother"]["optimize"] = "3"
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

def main():

    d = int(sys.argv[1])
    l = int(sys.argv[2])
    k = int(sys.argv[3])

    if(len(sys.argv)<=4):
        epsilons = [1.0, 50.0]
    else:
        epsilons = [float(sys.argv[4])]
    
    counter = 0

    preconditioners = ["diagonal"]

    for a in ["post", "symm"]:
    #for a in ["post"]:
        for o in range(1, 3):
            preconditioners.append("fdm_%s_%d_f" % (a, o));
        preconditioners.append("fdm_%s_v_f" % (a));

        #for o in range(1, 2):
        #    preconditioners.append("fdm-%d-r" % o);

    #for eps in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    for eps in epsilons:
        for solver in ["CG", "GMRES"]:
            preconditioners_to_be_used = preconditioners

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
  