import json
import os
from argparse import ArgumentParser

def run_instance(counter, l, input_file):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/" + input_file, 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["n refinements"] = l

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    
    counter = 0

    for l in range(0, 12):
        run_instance(counter, l, "large-scaling_diag.json")
        counter = counter + 1;

        run_instance(counter, l, "large-scaling_fdm1.json")
        counter = counter + 1;

        run_instance(counter, l, "large-scaling_fdm2.json")
        counter = counter + 1;

        run_instance(counter, l, "large-scaling_fdmv.json")
        counter = counter + 1;


if __name__== "__main__":
  main()
  