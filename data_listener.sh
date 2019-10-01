
inputs[0]=$1
inputs[1]=$2
inputs[2]=$3
inputs[3]=$4

inputs[4]=$5
inputs[5]=$6
inputs[6]=$7

a=1
while [ "$a" -eq 1 ]
do
    if [ -f "yodeling.txt" ]
    then
        a=0
    fi
    sleep 1
done

echo ${inputs[@]}
bash local_train_model_wrapper.sh ${inputs[@]}