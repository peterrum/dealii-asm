# mpirun -np 40 ./matrix_free_loop_08 matrix_free_loop_08.json | tee result.out
cat result.out | grep '>>' > result.out_1

cat result.out | grep "Memory read data volume"  | awk 'NR % 2 == 0' > result.out_2
cat result.out | grep "Memory write data volume" | awk 'NR % 2 == 0' > result.out_3

python ../experiments/matrix_free_loop_08_2.py result.out_1 result.out_2 result.out_3
