```
. ../experiments/smoother.sh
```

```
mkdir experiment-cartesian-6
cd experiment-cartesian-6
python ../../experiments/cartesian.py 3 5 6
array=($(ls *.json))
mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
cd ..
```

```
mkdir experiment-ball-6
cd experiment-ball-6
python ../../experiments/ball.py 3 4 6
array=($(ls *.json))
mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
cd ..
```

```
mkdir experiment-kershaw
cd experiment-kershaw
python ../../experiments/kershaw.py 3 3 6
array=($(ls *.json))
mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
cd ..
```

```
mkdir experiment-anisotropy
cd experiment-anisotropy
python ../../experiments/anisotropy.py 3 5 6
array=($(ls *.json))
mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
cd ..
```

### jk

```
mkdir experiment-patch-sequence
cd experiment-patch-sequence
likwid-mpirun -np 40 -f -g CACHES -m -O ../matrix_free_loop_08 ../../experiments/matrix_free_loop_08_sequence.json | tee result_caches.out

```

```
mkdir experiment-patch-degree
cd experiment-patch-degree
python ../../experiments/matrix_free_loop_08_degree.py
array=($(ls *.json))
likwid-mpirun -np 40 -f -g CACHES -m -O ../matrix_free_loop_08 "${array[@]}"   | tee result_caches.out
likwid-mpirun -np 40 -f -g FLOPS_SP -m -O ../matrix_free_loop_08 "${array[@]}" | tee result_flops.out

```

```
mkdir experiment-patch-cheby
cd experiment-patch-cheby
python ../../experiments/matrix_free_loop_08_cheby.py
array=($(ls *.json))
likwid-mpirun -np 40 -f -g CACHES -m -O ../matrix_free_loop_08 "${array[@]}"   | tee result_caches.out
likwid-mpirun -np 40 -f -g FLOPS_SP -m -O ../matrix_free_loop_08 "${array[@]}" | tee result_flops.out

```