import json
import os
import sys

def run_instance(counter, d, l, mg_sequence,input_file):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/" + input_file, 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["dim"]           = d
    datastore["n refinements"] = l
    datastore["preconditioner"]["mg type"] = mg_sequence

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    d           = int(sys.argv[1])
    mg_sequence = sys.argv[2] if len(sys.argv) > 2 else "ph"
    
    counter = 0

    for l in range(0, 12):
        run_instance(counter, d, l, mg_sequence, "large-scaling-opt_diag.json")
        counter = counter + 1;

        run_instance(counter, d, l, mg_sequence, "large-scaling-opt_fdm1.json")
        counter = counter + 1;

        run_instance(counter, d, l, mg_sequence, "large-scaling-opt_fdm2.json")
        counter = counter + 1;

        run_instance(counter, d, l, mg_sequence, "large-scaling-opt_fdmv.json")
        counter = counter + 1;


if __name__== "__main__":
  main()
  
