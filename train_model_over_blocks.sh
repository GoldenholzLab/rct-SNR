
python main_python_scripts/initialize_model.py

num_blocks=20

for ((block_num=0; block_num<=$num_blocks; block_num=block_num+1))
do
    #echo $block_num
    python main_python_scripts/train_model.py 100 $block_num
done