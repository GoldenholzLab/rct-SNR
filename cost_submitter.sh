

num_estims=10

for ((x=1; x<=$num_estims; x=x+1))
do
    bash RR50_or_MPC_cost_wrapper.sh RR50 smart $x
    bash RR50_or_MPC_cost_wrapper.sh MPC  smart $x
    bash TTP_cost_wrapper.sh              smart $x
    bash RR50_or_MPC_cost_wrapper.sh RR50 dumb  $x
    bash RR50_or_MPC_cost_wrapper.sh MPC  dumb  $x
    bash TTP_cost_wrapper.sh              dumb  $x
done
