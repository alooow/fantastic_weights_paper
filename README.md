# Code for the Paper: *Fantastic Weights and How to Find Them: Where to Prune in Dynamic Sparse Training*
[Link to paper](https://arxiv.org/abs/2306.12230)



## Repo Structure

`data_handling` - all code for downloading & loading the data  
`models` - the models.  
`singularity` - definition of the used contained (optional)  
`specs` - directory with different experiments settings with which the model has been run. 
`main.py` - the main file.  
`trainer.py` - the logic for training and growing    
`sparselearning` - the logic behind sparse training (main mask logic in `core.py`)  
`ImageNet` - the logic behind sparse training for ImageNet on Wide ResNet, not used(main mask logic in `core.py`)  

## Running Experiments

### Base Setup

(Best for running locally or just testing the code)

Create a conda environment according to `environment.yml`:
```
conda env create -f environment.yml
```

In addition, please install [W&B](https://wandb.ai/site) within the environment:
```
conda activate sparse
conda install wandb -c conda-forge   # or pip install wandb
```
If you are planning to use the logging to W&B, set the W&B environment variables:

```
export WANDB_ENTITY_NAME=<your_entity_name>
export WANDB_PROJECT_NAME=<your_project_name>
export WANDB_API_KEY=<your_API_key>
```
Please make sure that pytorch with cuda is correctly installed. The pytorch and cudatoolkit versions in the environment specification file are the ones used in our experiments. If such configuration is not available on your system, use the latest versions of the packages.  

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

**Note** This solution will not allow you to run all the experiments simultaneously as separate jobs. You will need to manually implement all loops and hyperparameter setups based on the hyperparametrs information in the `specs` folder. Therefore, this is a good option for testing but not for reproducing all our experiments. 

**Note** If you plan to use our singularity container (recommended for reproducibility), you don't need to create the environment since everything is already provided.  

### With singularity, no mrunner :

1. Install singularity: https://docs.sylabs.io/guides/3.0/user-guide/installation.html
2. Download the provided by us ready-to use container from [zendoo](https://zenodo.org/record/7965582). Alternatively, you can build the container on your own using the `/singularity/sparse.def` file:
``` 
sudo singularity build sparse.sif sparse.def
```
For more information on building containers see https://docs.sylabs.io/guides/3.0/user-guide/build_a_container.html.

3. Run the code within the container (locally):

```
singularity exec -H $PWD:/homeplaceholder sparse.sif <command>
```

where "<command>" is the command that you want to run, for instance:
```
singularity exec -H $PWD:/homeplaceholder sparse.sif python main.py --use_wandb false --data cifar10 --model conv_cifar10
```

If you have GPU support, you may want to add the `--nv` option after the `-H $PWD:/homeplaceholder`. Note that the code looks for the
cifar10 dataset under the `CIFAR10` environment variable. You may pass the variable using 
`--env CIFAR10=<your_path>` after `-H $PWD:/homeplaceholder`.  If you want to include more environment variables (for instance the WANDB_API_KEY etc.) pass them comma-separated in the `--env` option. You may also need to bind the directory with the dataset:

```
singularity exec -H $PWD:/homeplaceholder --bind /dataset/cifar10:/dataset/cifar10 --nv --env CIFAR10=/dataset/cifar10,WANDB_API_KEY=<your_key> sparse.sif python main.py --use_wandb false --data cifar10 --model conv_cifar10
```

**Note**: we used singularity3.0 to build the container. If that version is no longer supported, please use the latest (as of May 2023) Apptainer: https://apptainer.org/. It should link the `singularity` command to `apptainer` command and proceed as above. 


### With singularity and mrunner (Recommended):

1. Install singularity on the host. Build or Download the container to the host as in the instructions above.
2. Create an environment locally (can be without cuda) and install mrunner locally within that environment: `pip install git+https://gitlab.com/awarelab/mrunner.git`. 
By default mrunner may look for a `"NEPTUNE_PROJECT_NAME"` and `"NEPTUNE_API_TOKEN"` environment variables, which are used to connect the logging to neptune.ai. 
If the code below will ask you to provide such variables, just set them to empty strings.  **NOTE**: The environment is only needed for the mrunner to parse the grid search arguments on your local side to prepare the experiment specifications for each hyperparameter configuration. Then the repository is copied to the host machine (see `storage_dir` below), creating a separate directory for each hyperparameter configuration. Then the code is run on the host machine using singularity.
3. Prepare an experiment specification file (or use any of the files from the `specs` directory. See `specs/README_experiments.md`)
4. Prepare a context definition that contains the information needed to run a job on a server. It uses [slurm](https://slurm.schedmd.com/documentation.html)    
To do so, create a `config.yaml` file with the following structure:
```
contexts:
  mycontext:
    account: <your slurm account>
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
- `account` - name of your slurm account (the `-A` option of slurm). If you have a default one set up, you can remove this line. 
- `partition` - the `-p` option of slurm. Write here the partition you want to use.   
- `slurm_url` - the address used to login to the host (typically username@hostname).  
- `storage_dir` - mrunner will copy this repository to the host. If the experiment file contains more than one experiment, then each will be assigned a seprate directory. (This is why it is **important** to set the dataset as an environment variable, so that it is shared by all the experiments). 
- `singularity_container` - any arguments that need to be passed to singularity 
- `<path-to-container>` - path to the `sparse.sif` on the host. (You need to copy the .sif to the host or create it at the host machine).

The context definition supports other `slurm` commands (for instance `time`, etc.)
**NOTE!!!*** Remember to set your `ssh` configuration to use private keys and add the key using `ssh-add` 

To run any of the scripts (see `specs/README_experiments.md`) use:

```
mrunner --config config.yaml --context mycontext run specs/<spec_name>
```

For example:

```
mrunner --config config.yaml --context mycontext run specs/mlp/001c_dst.py 
```

The command above will submit each of the 550 experiments specified by the grid in the `specs/mlp/001c_dst.py` file as a separate slurm jobs. The code will be copied to separate folders created for each of the 550 jobs under the  `<storage_dir_on_host>/mrunner_scratch/weight_importance/<random-generated_string>/<random-generated_string>_<job_number>` path. It is, therefore, important not to store or copy large files from the local code. See `specs/README_experiments.md` for examples on how to exclude large files from copying and how to share the datasets by using an environmental variable. 


### With mrunner, no singularity (Not Recommended):

Install mrunner in the environment:

```
pip install git+https://gitlab.com/awarelab/mrunner.git
```

Note that by default mrunner may look for a `"NEPTUNE_PROJECT_NAME"` and `"NEPTUNE_API_TOKEN"` environment variables, which are used to connect the logging to neptune.ai. 
If the code below will ask you to provide such variables just set them to empty strings. 

1. Prepare an experiment specification. Examples of such files can be found in `specs`

### Locally
2.  Activate the `sparse` conda environment (created in previous section) and run:

```
python mrun.py --ex <spec_name>
```

where `<spec_name>` is the filepath of the experiment file (for instance `/specs/mlp/001a_dense.py`).

### On the Host

2. Prepare the conda environment on the host.
3. Prepare the context file as in the previous section, but instead of `singularity container` use `conda`

```
contexts:
  mycontext:
    account: <your slurm account>
    backend_type: slurm
    cmd_type: sbatch
    partition: <partition-name>
    nodes: 1
    gpu: 1
    cpu: 4
    mem: 4G
    slurm_url: <username@hostname>
    storage_dir: <storage_dir_on_host>
    conda: <name-of-the-conda-env-on-the-host>
```

4. Run the command:

```
mrunner --config config.yaml --context mycontext run specs/<spec_name>
```



# Experiment Setups:

All setups for the experiments conducted in the main text are available in the specifications under directory `specs`. Please see `specs/README_experiments.md` for a summary.

For the ImageNet, look at the `scripts/ImageNet` files (note that they need to be run within the singularity environment, if you are using one. Otherwise, set the conda environment in the script).
To Run ImageNet with DST use:
```
sbatch scripts/ImageNet/resnet50_dst_b32 10001 <pruning criterion> 1
```
The first number is the master port, the second is the name of the pruning criterion (one of `["MEST", "SET", "magnitude", "SNIP", "ReciprocalSensitivity", "Random"]`,), the last one is the seed. See `imagenet_main.py --help` for other options. 


# Acknowledgements

The code is based on the repositories:

[https://github.com/VITA-Group/Random_Pruning](https://github.com/VITA-Group/Random_Pruning)

and

[https://github.com/TimDettmers/sparse_learning](https://github.com/TimDettmers/sparse_learning)

The LICENCE of the above-mentioned `sparse_learning` repository is in `sparselearning/original_sparselearning_LICENSE`.





