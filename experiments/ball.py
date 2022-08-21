import json
import os
from argparse import ArgumentParser

def run_instance(counter, d, l, k, preconditioner, sequence, s):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["name"]          = preconditioner
    datastore["mesh"]["name"]  = "hyperball"
    datastore["dim"]           = d
    datastore["n refinements"] = l
    datastore["degree"]        = k

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
        datastore["preconditioner"]["mg smoother"]["preconditioner"]["n overlap"] = props[1]

        if props[2] == "f":
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

    #for o in range(1, k+1):
    #    preconditioners.append("fdm-%d-f" % o);

    for o in range(1, k+1):
        preconditioners.append("fdm-%d-r" % o);


    for preconditioner in preconditioners:
        for sequence in ["bisect", "go to one", "decrease by one"]:
            for s in range(1, 6):
                run_instance(counter, d, l, k, preconditioner, sequence, s)
                counter = counter + 1;


if __name__== "__main__":
  main()
  