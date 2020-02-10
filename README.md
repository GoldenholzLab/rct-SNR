
This source code for this repository consists of bash scripts which make calls to other Python 3 scripts. 
The Python 3 scripts are placed in two separate folders: 'main_python_scripts' and 'utility_code', while 
the bash scripts are placed in the top-level directory of this repository. 

In order to execute the code within this repository as is, it is not necessary to call on any of the python 
scripts, simply executing the bash scripts should do. The two following workflows describe which bash scripts
to call.

1) Generation of data for training and testing of deep learning models

    Data has to be generated on an HPC (High Parallel Computing) cluster in order to train a bunch of deep learning models.
    This project utilized the O2 cluster supported by the Research Computing group at Harvard Medical School.

    For the O2 cluster, the following command was used: 'sbatch submit_keras_data_generation_wrappers.sh'.

    This command ultimately results in the creation of many files contained within a folder specified by the 'submit_keras_data_generation_wrappers.sh' script.

2) Actual training and testing of deep learning models aloing with generation of results

    Once all the data has been generated, then the following command can be executed: 'bash train_model_over_blocks.sh'

    This command should start training deep learning models and ultimately result in the creation of four figures. Several 
    intermediate JSON and text files will also be created. These files store data which is necessary for the creation of these
    figures, so they should remain unchanged until the process of Figure generation is complete.

    This step does not require an HPC cluster.

