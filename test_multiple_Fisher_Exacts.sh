for ((i=1; i<16; i=i+1));
do
    sbatch test_Fisher_Exact.sh $i
done