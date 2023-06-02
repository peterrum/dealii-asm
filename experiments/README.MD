### Section 2: FDM tests (on gold)

```
mkdir experiment-patch-sequence
cd experiment-patch-sequence
likwid-mpirun -np 40 -f -g CACHES -m -O ../matrix_free_loop_08 ../../experiments/matrix_free_loop_08_sequence.json | tee result_caches.out
cd ..


mkdir experiment-patch-stat
cd experiment-patch-stat
likwid-mpirun -np 40 -f -g CACHES -m -O ../matrix_free_loop_08 ../../experiments/matrix_free_loop_08_stat.json | tee result_caches.out
cd ..


mkdir experiment-patch-stat-b
cd experiment-patch-stat-b
likwid-mpirun -np 40 -f -g CACHES -m -O ../matrix_free_loop_08 ../../experiments/matrix_free_loop_08_stat_b.json | tee result_caches.out
cd ..


mkdir experiment-patch-stat-c
cd experiment-patch-stat-c
likwid-mpirun -np 40 -f -g CACHES -m -O ../matrix_free_loop_08 ../../experiments/matrix_free_loop_08_stat_c.json | tee result_caches.out
cd ..

```

```
mkdir experiment-patch-degree
cd experiment-patch-degree
python ../../experiments/matrix_free_loop_08_degree.py
array=($(ls *.json))
likwid-mpirun -np 40 -f -g CACHES -m -O ../matrix_free_loop_08 "${array[@]}"   | tee result_caches.out
likwid-mpirun -np 40 -f -g FLOPS_SP -m -O ../matrix_free_loop_08 "${array[@]}" | tee result_flops.out

```

### Section 3: Chebyshev tests (on gold)

```
mkdir experiment-patch-cheby
cd experiment-patch-cheby
python ../../experiments/matrix_free_loop_08_cheby.py
array=($(ls *.json))
likwid-mpirun -np 40 -f -g CACHES -m -O ../matrix_free_loop_08 "${array[@]}"   | tee result_caches.out
likwid-mpirun -np 40 -f -g FLOPS_SP -m -O ../matrix_free_loop_08 "${array[@]}" | tee result_flops.out

```

```
mkdir experiment-patch-cheby-degree
cd experiment-patch-cheby-degree
python ../../experiments/matrix_free_loop_08_cheby_degree.py
array=($(ls *.json))
mpirun -np 40 ../matrix_free_loop_08 ./input_0000.json | tee result_cartesian.output
mpirun -np 40 ../matrix_free_loop_08 ./input_0001.json | tee result_general.output

```

```
mkdir -p experiment-multigrid_cheby_degree
cd experiment-multigrid_cheby_degree
python ../../experiments/multigrid_cheby_degree.py
array=($(ls *.json))
mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
cd ..
```

### Section 4: Multigrid tests (on gold)

#### Full

```
. ../experiments/anisotropy.sh
```

```
. ../experiments/kershaw.sh
```

```
. ../experiments/ball.sh
```

#### Short

```
. ../experiments/anisotropy.sh 1

cd experiment-anisotropy-4/
array=($(ls *.json))
mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
cd ..

cd experiment-anisotropy-7/
array=($(ls *.json))
mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
cd ..
```

```
. ../experiments/kershaw.sh 1

cd experiment-kershaw-4/
array=($(ls *.json))
mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
cd ..

cd experiment-kershaw-7/
array=($(ls *.json))
mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
cd ..
```

```
. ../experiments/ball.sh
```

### Section 4: Multigrid tests (on SuperMUC-NG)

```
mkdir experiments-large-scaling-opt
cd experiments-large-scaling-opt
python ../../experiments/large-scaling-create-job-files.py
python ../../experiments/large-scaling-opt.py 3
array=($(ls *.json))
```
