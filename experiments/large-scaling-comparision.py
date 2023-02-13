import json
import os
import sys

def run_instance(counter, d, geometry, l, cycle, input_file):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/" + input_file, 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["dim"]             = d
    datastore["n refinements"]   = l
    datastore["mesh"]["name"]    = geometry

    cycle = cycle.split("-")
    datastore["preconditioner"]["mg type"]         = cycle[0]
    datastore["preconditioner"]["n coarse cycles"] = cycle[1]

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():

    d        = sys.argv[1]
    geometry = sys.argv[2]
    
    counter = 0

    cycles = ["hp-1", "ph-1", "ph-2", "ph-5", "ph-10", "p-1", "p-2", "p-5", "p-10"]

    for l in range(0, 12):
        for cycle in cycles:
            run_instance(counter, d, geometry, l, cycle, "large-scaling_diag.json")
            counter = counter + 1;

        for cycle in cycles:
            run_instance(counter, d, geometry, l, cycle, "large-scaling_fdm1.json")
            counter = counter + 1;

        for cycle in cycles:
            run_instance(counter, d, geometry, l, cycle, "large-scaling_fdm2.json")
            counter = counter + 1;

        for cycle in cycles:
            run_instance(counter, d, geometry, l, cycle, "large-scaling_fdmv.json")
            counter = counter + 1;


if __name__== "__main__":
  main()
  