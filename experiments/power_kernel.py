import json
import os
from argparse import ArgumentParser

def run_instance(counter, d, k, s, g, v):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/power_kernel.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["dim"]              = d
    datastore["fe degree"]           = k
    datastore["n subdivisions"]   = s
    datastore["cell granularity"] = g
    datastore["n lanes"]          = v

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")

    parser.add_argument('d', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('s', type=int)
    
    arguments = parser.parse_args()
    return arguments

def main():
    options = parseArguments()

    d = options.d
    k = options.k
    s = options.s
    
    counter = 0

    for g in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 10000000]:
        for v in [8]:
            run_instance(counter, d, k, s, g, v)
            counter = counter + 1;


if __name__== "__main__":
  main()
  
