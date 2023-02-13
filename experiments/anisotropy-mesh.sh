executation_type=0

if [[ $# -eq 1 ]] ; then
  executation_type=$1
fi

for i in "0 1" "1 2" "2 5" "3 10" "4 20"
do
    set -- $i

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "1" ]; then
        mkdir -p experiment-anisotropy-4-$1
    fi

    cd experiment-anisotropy-4-$1

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "1" ]; then
        python ../../experiments/anisotropy.py 3 6 4 $2
    fi

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "2" ]; then
        array=($(ls *.json))
        mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
    fi
    cd ..
done

