import json
import os
from argparse import ArgumentParser

def run_instance(counter, type, degree_and_subdivisions):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/matrix_free_loop_08_cheby.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["fe degree"]             = degree_and_subdivisions[0]
    datastore["n subdivisions"]        = degree_and_subdivisions[1]
    datastore["preconditioner types"]  = " ".join(type)

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():    
    counter = 0

    type = ["cheby-3-0-diag",         "cheby-3-3-diag", 
            "cheby-3-0-symm-1-c",     "cheby-3-2-symm-1-c", 
            "cheby-3-0-symm-2-g-p-n", "cheby-3-2-symm-2-g-p-n", 
            "cheby-3-0-symm-v-c",     "cheby-3-2-symm-v-c"
           ]

    for degree_and_subdivisions in [[2, 45], [3, 44], [4, 40], [5, 39], [6, 38], [7, 36]]:
        run_instance(counter, type, degree_and_subdivisions)
        counter = counter + 1;


if __name__== "__main__":
  main()
  