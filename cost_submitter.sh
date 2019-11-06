

num_estims=10

for ((x=1; x<=$num_estims; x=x+1))
do
    bash cost_wrapper.sh RR50 smart $x
    bash cost_wrapper.sh MPC  smart $x
    bash cost_wrapper.sh TTP  smart $x
    bash cost_wrapper.sh RR50 dumb  $x
    bash cost_wrapper.sh MPC  dumb  $x
    bash cost_wrapper.sh TTP  dumb  $x
done
