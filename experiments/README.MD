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