import json
import os
from argparse import ArgumentParser

def run_instance(counter, d, l, input_file):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/" + input_file, 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["dim"]           = d
    datastore["n refinements"] = l

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")

    parser.add_argument('d', type=int)
    
    arguments = parser.parse_args()
    return arguments

def main():
    options = parseArguments()

    d = options.d
    
    counter = 0

    for l in range(0, 12):
        run_instance(counter, d, l, "large-scaling-ras_0.json")
        counter = counter + 1;

        run_instance(counter, d, l, "large-scaling-ras_1.json")
        counter = counter + 1;

        run_instance(counter, d, l, "large-scaling-ras_2.json")
        counter = counter + 1;


if __name__== "__main__":
  main()
  