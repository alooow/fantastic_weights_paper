# Weight Importance


## Repo Structure

`data_handling` - all code for downloading & loading the data  
`models` - the models.  
`singularity` - definition of the used contained (optional)  
`specs` - directory with different experiments settings with which the model has been run. 
`main.py` - the main file.  
`trainer.py` - the logic for training and growing    
`sparselearning` - the logic behind sparse training (main mask logic in `core.py`)  
`ImageNet` - the logic behind sparse training for ImageNet on Wide ResNet, not used(main mask logic in `core.py`)  

## Run


### With singularity and mrunner (recommened):

1. Install singularity and mrunner:
```
pip install git+https://gitlab.com/awarelab/mrunner.git
```
For instructions on how to install singularity see https://docs.sylabs.io/guides/3.0/user-guide/installation.html and chapter "With singularity, no mrunner (or mrunner locally) (Not Recommended)" below.    
2. Prepare an experiment specification file (you can use any of the files in the `specs` directory as a reference. See `specs/README_experiments.md`)     
3. Prepare a context definition which contains the information needed to run a job on a server. It uses slurm (https://slurm.schedmd.com/documentation.html)    
To do so, create an `config.yaml` file with the following structure:
```
contexts:
  mycontext:
    backend_type: slurm
    cmd_type: sbatch
    partition: <partition-name>
    nodes: 1
    gpu: 1
    cpu: 4
    mem: 4G
    slurm_url: <username@hostname>
    storage_dir: <storage_dir_on_host>
    singularity_container: -H $PWD:/homeplaceholder --bind <your_binds>  --nv --env <all_env_variables> -B $TMPDIR:/tmp <path-to-container>

```
where:
- `mycontext` - name of the context, can be anything   
- `partition` - the `-p` option of slurm. Write here the partition you want to use.   
- `slurm_url` - the address used to login to the host (typically username@hostname).  
- `storage_dir` - mrunner will copy this repository to the host. If the experiment file contains more than one experiment, then each will be assigned a seprate directory. (This is why it is **important** to set the dataset as an environment variable, so that it is shared by all the experiments -- see `specs/README_experiments.md` or the chapters below for example on hot to do that). 
- `singularity_container` - any arguments that need to be passed to singularity 
- `<path-to-container>` - path to the `sparse.sif` on the host. (You need to copy the .sif to the host or create it at the host machine).

The context definition supports other `slurm` commands (for instance `account`, `time`, etc.)
In addition, remember to set your `ssh` configuration to use private keys and add the key using `ssh-add` 

To run any of the scripts (see `specs/README_experiments.md`) use:

```
mrunner --config config.yaml --context mycontext run specs/<spec_name>
```

For example:

```
mrunner --config config.yaml --context mycontext run specs/mlp/001c_dst.py 
```

**Note** The command above will submit each of the 550 experiments specified by the grid in the `specs/mlp/001c_dst.py` file as a separate slurm jobs. The code will be copied to a separate folder created for each of the 550 jobs under the  `<storage_dir_on_host>/mrunner_scratch/weight_importance/<random-generated_string>/<random-generated_string>_<job_number>` path. It is therefore important not to store or copy large files from the local code. See `specs/README_experiments.md` for examples on how to exlude large files from copying and how to share the datasets by using an environmental variable. 

### Without Singularity and Mrunner (Not Recommended)

Create a conda environment according to `environment.yml`:
```
conda env create -f environment.yml
```
In addition, install W&B within the environment:
```
conda activate sparse
conda install wandb -c conda-forge   # or pip install wandb
```

Please make sure that pytorch with cuda is correctly installed. 
Activate the created  environment and run:
```
export CIFAR10_PYTORCH="./data"
python main.py --use_wandb false --data cifar10 --model conv_cifar10
```
to train a simple (baseline) MLP model on CIFAR10. See `main.py --help`  for more command line options. 
The datasets are expected to be found under the following names:
- `CIFAR10_PYTORCH` - CIFAR10
- `TABULAR_DATA_DIR` - Higgs dataset
- `CIFAR100_PYTORCH` - CIFAR100
- `TINY_IMAGENET` - Tiny-ImageNet
- `FASHION_MNIST_PYTORCH` - FashionMNIST
- `IMAGENET_PYTORCH` - ImageNet

To use wandb:
```
export WANDB_ENTITY_NAME=<your_entity_name>
export WANDB_PROJECT_NAME=<your_project_name>
export WANDB_API_KEY=<your_API_key>
```
**Note** Using this solution will not allow you to run all the experiments at once as separate jobs. You will need to implement all loops and hyperparameter setups manually based on the information on hyperparametrs in the `specs` folder. Therefore this is a good option for testing, but not for reproducing all our experiments. 


### With mrunner (no singularity, locally) (Not Recommended):

Install mrunner in the environment:

```
pip install git+https://gitlab.com/awarelab/mrunner.git
```

Note that by default mrunner may look for a `"NEPTUNE_PROJECT_NAME"` and `"NEPTUNE_API_TOKEN"` environment variables, which are used to connect the logging to neptune.ai. 
If the code below will ask you to provide such variables just set them to empty strings. 

1. Prepare an experiment specification. Examples of such files can be found in `specs`

2.  Activate the `sparse` conda environment (created in previous section) and run:

```
python mrun.py --ex <spec_name>
```


where `<spec_name>` is the filepath of the experiment file (for instance `/specs/mlp/001a_dense.py`).

### With singularity, no mrunner (or mrunner locally) (Not Recommended):

1. Install singularity: https://docs.sylabs.io/guides/3.0/user-guide/installation.html
2. Build the container using the `/singularity/sparse.def` file: 
``` 
sudo singularity build sparse.sif sparse.def
```
3. Run the code within the container (locally):

```
singularity exec -H $PWD:/homeplaceholder sparse.sif <command>
```

where "<command>" is any of the commands from sections "Without Singularity and Mrunner" or "With mrunner locally". For instance:
```
singularity exec -H $PWD:/homeplaceholder sparse.sif python mrun.py --ex <spec_name>
```

If you have GPU support, you may want to add the `--nv` option after the `-H $PWD:/homeplaceholder`. Note that the code looks for the
cifar10 dataset under the `CIFAR10` environment variable. You may pass the variable using 
`--env CIFAR10=<your_path>` after `-H $PWD:/homeplaceholder`. You may then need to bind the directory with the dataset:

```
singularity exec -H $PWD:/homeplaceholder --bind /dataset/cifar10:/dataset/cifar10 --nv --env CIFAR10=/dataset/cifar10 sparse.sif python mrun.py --ex <spec_name>
```

**Note**: we used singularity3.0 to build the container. If that version is no-longer supported, please use the latest (as of May 2023) Apptainer: https://apptainer.org/. It should link the `singularity` command to `apptainer` command and proceed as above. 


# Experiment Setups:

All setups for the experiments conducted in the main text are available in the specifactions under directory `specs`.
For the ImageNet, look at the `scripts/ImageNet` files (note that they need to be run within the singularity environment, if you are using one. Otherwise set the conda environment in the script).
To Run ImageNet with DST use:
```
sbatch scripts/ImageNet/resnet50_dst_b32 10001 <pruning criterion> 1
```
The first number is the master port, the second is the name of the pruning criterion (one of `["MEST", "SET", "magnitude", "SNIP", "ReciprocalSensitivity", "Random"]`,), the last one is the seed. See `imagenet_main.py --help` for other options. 


# Note

The code is based on the repositories:

[https://github.com/VITA-Group/Random_Pruning](https://github.com/VITA-Group/Random_Pruning)

and

[https://github.com/TimDettmers/sparse_learning](https://github.com/TimDettmers/sparse_learning)

The LICENCE of the above mentioned `sparse_learning` repository is in `sparselearning/original_sparselearning_LICENSE`.





