
"" > temp

./reduced_access_02 4   0 0 0 0 0 0 0 0 0 0 0 0   0 0 0 0 0 0 | tee -a temp

./reduced_access_02 4   1 0 0 0 0 0 0 0 0 0 0 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 1 0 0 0 0 0 0 0 0 0 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 1 0 0 0 0 0 0 0 0 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 1 0 0 0 0 0 0 0 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 1 0 0 0 0 0 0 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 1 0 0 0 0 0 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 1 0 0 0 0 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 0 1 0 0 0 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 0 0 1 0 0 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 0 0 0 1 0 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 0 0 0 0 1 0   0 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 0 0 0 0 0 1   0 0 0 0 0 0 | tee -a temp

./reduced_access_02 4   0 0 0 0 0 0 0 0 0 0 0 0   1 0 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 0 0 0 0 0 0   0 1 0 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 0 0 0 0 0 0   0 0 1 0 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 0 0 0 0 0 0   0 0 0 1 0 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 0 0 0 0 0 0   0 0 0 0 1 0 | tee -a temp
./reduced_access_02 4   0 0 0 0 0 0 0 0 0 0 0 0   0 0 0 0 0 1 | tee -a temp


diff ../reduced_access_02.result temp
