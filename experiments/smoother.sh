mkdir -p experiment-smoother

mpirun -np 40 ./matrix_free_loop_02 3 4 6 1 1 1 0 | tee experiment-smoother/data_cartesian_vmult.out
mpirun -np 40 ./matrix_free_loop_02 3 4 6 1 0 1 0 | tee experiment-smoother/data_cartesian_step.out

mpirun -np 40 ./matrix_free_loop_02 3 4 6 1 1 0 0 | tee experiment-smoother/data_deformed_vmult.out
mpirun -np 40 ./matrix_free_loop_02 3 4 6 1 0 0 0 | tee experiment-smoother/data_deformed_step.out


