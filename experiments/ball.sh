executation_type=0

if [[ $# -eq 1 ]] ; then
  executation_type=$1
fi

for i in "2 5" "3 5" "4 4" "5 4" "6 4"
do
    set -- $i

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "1" ]; then
        mkdir -p experiment-ball-$1
    fi

    cd experiment-ball-$1

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "1" ]; then
        python ../../experiments/ball.py 3 $2 $1
    fi

    if [ "$executation_type" -eq "0" ] || [ "$executation_type" -eq "2" ]; then
        array=($(ls *.json))
        mpirun -np 40 ../element_centered_preconditioners_01 "${array[@]}" | tee output.out
    fi
    cd ..
done

