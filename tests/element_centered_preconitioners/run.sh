#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# loop over all json files in folder
for f in $SCRIPT_DIR/$1/*.json
do
  echo -n "Processing " $f

  # run program
  ./element_centered_preconditioners_01 $f > __temp__

  # create diff of output
  grep -v -x -F "DEBUG!" __temp__ > __filtered__
  diff ${f::-5}.output __filtered__ > __diff__

  # if diff is empty everything is fine
  if [ -s __diff__ ]; then
    printf ": \033[0;31mfailed\033[0m\n"
  else
    printf ": \033[0;32msuccess\033[0m\n"
  fi

  cp __filtered__ ${f::-5}.output
done