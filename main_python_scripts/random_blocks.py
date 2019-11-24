import sys
import random

if(__name__=='__main__'):

    num_total_blocks    = int(sys.argv[1])
    num_testing_blocks = int(sys.argv[2])
    num_training_blocks  = num_total_blocks - num_testing_blocks

    random_training_blocks = random.sample(range(1, num_total_blocks + 1), num_training_blocks)
    random_training_blocks.sort()

    random_training_blocks_str = ''
    for training_block_index in range(num_training_blocks):
        random_training_blocks_str = random_training_blocks_str + str(random_training_blocks[training_block_index]) + ' '

    print(random_training_blocks_str)

    '''
    leftout_testing_blocks = []
    for block_index in range(1, num_total_blocks + 1):

        if(random_training_blocks.count(block_index) == 0):

            leftout_testing_blocks.append(block_index)

    print(random_training_blocks)
    print(leftout_testing_blocks)
    '''
