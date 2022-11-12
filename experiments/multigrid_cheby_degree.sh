executation_type=0

if [[ $# -eq 1 ]] ; then
  executation_type=$1
fi

for i in "2 6" "3 6" "4 6" "5 6" "6 5"
do
    set -- $i

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "1" ]; then
        mkdir -p experiment-anisotropy-$1
    fi

    cd experiment-anisotropy-$1

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "1" ]; then
        python ../../experiments/anisotropy.py 3 $2 $1
    fi

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "2" ]; then
        array=($(ls *.json))
        mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
    fi
    cd ..
done

