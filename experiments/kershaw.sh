executation_type=0

if [[ $# -eq 1 ]] ; then
  executation_type=$1
fi

for i in "2 4" "3 4" "4 3" "5 3" "6 3"
do
    set -- $i

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "1" ]; then
        mkdir -p experiment-kershaw-$1
    fi

    cd experiment-kershaw-$1

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "1" ]; then
        python ../../experiments/kershaw.py 3 $2 $1
    fi

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "2" ]; then
        array=($(ls *.json))
        mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
    fi
    cd ..
done

