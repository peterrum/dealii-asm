import json
import os
from argparse import ArgumentParser

def run_instance(counter, type, degree_and_subdivisions, use_cartesian_mesh):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/matrix_free_loop_08_cheby.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["fe degree"]             = degree_and_subdivisions[0]
    datastore["n subdivisions"]        = degree_and_subdivisions[1]
    datastore["n subdivisions"]        = degree_and_subdivisions[1]
    datastore["use cartesian mesh"]    = use_cartesian_mesh
    datastore["preconditioner types"]  = " ".join(type)

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():    
    counter = 0

    type_reference = ["cheby-%d-0-diag",         "cheby-%d-3-diag", 
            "cheby-%d-0-symm-1-c",     "cheby-%d-2-symm-1-c", 
            "cheby-%d-0-symm-2-g-p-n", "cheby-%d-2-symm-2-g-p-n", 
            "cheby-%d-0-symm-v-c",     "cheby-%d-2-symm-v-c"
           ]

    type = [[i % degree for i in type_reference] for degree in range(1, 6)]

    types = []

    for t in type:
      types = types + t

    run_instance(counter, types, [4, 40], True)
    counter = counter + 1;

    run_instance(counter, types, [4, 40], False)
    counter = counter + 1;


if __name__== "__main__":
  main()
  