# Experimental Setup

This folder contains the summary of the hyperparameters and setups used to run the experiments from the main paper. 
Each experiment setup consisted up to several specifiaction (`spec`) files (standard python extension). Those files 
describe how many experiments and with what setup to run (as required by mrunner). To run locally the specification use the `mrun.py` file.
For example:
```
python mrun.py --ex specs/cfashion_mnist/900c_dst.py
``` 

**NOTE**: Each file containes multiple runs. For instance, the above line will try to sequentially run 550 experiments! This is inefficient and we advise to either use the files with actual 
mrunner or only used them as reference for your own scripts. To run with mrunner and multiple jobs use the following command:

```
mrunner --config <path-to-your-config-file.yaml> --context <your_context_name> run specs/cfashion_mnist/900c_dst.py
``` 

 It will sent each of the 550 experiments as a separate job to the server. You have to prepare the config file on your own. It should have the following structure:

```
contexts:
  <your_context_name>:
    account: <your_slurm_account_name>
    partition: <your_slurm_partition_name>
    backend_type: slurm
    cmd_type: sbatch
    cpu: 5
    mem: 8G
    gpu: 1
    nodes: 1
    ntasks: 1
    slurm_url: <your_server_address>
    storage_dir: <path_on_your_server_where_the_code_will_be_copied>
    singularity_container: -H $PWD:/homeplaceholder --bind <your_binds> --nv --env WANDB_API_KEY=<your_api_key>,WANDB_ENTITY_NAME=<your_entity_name>,WANDB_PROJECT_NAME=<your_project_name>,CIFAR10_PYTORCH=<your-data-path>/cifar10_pytorch,MNIST_PYTORCH=<your-data-path>/mnist_pytorch,CIFAR100_PYTORCH=<your-data-path>/cifar100_pytorch -B $TMPDIR:/tmp <path-to-container>

```

In the above configuration file:
- <your_context_name> - a short name to describe your context. Can be anything, e.g. shot version of the server name etc.
- <your_slurm_account_name> - your slurm account name if one is not set as a default by your server (https://slurm.schedmd.com/sacctmgr.html)
- <your_slurm_partition_name> - yout slurm partition name if one is not set as default by your server.
- <your_server_address> - the address of your server (used for log-in). **Be sure to use passphrase and ssh-add yout key**. 
- <path_on_your_server_where_the_code_will_be_copied> - mrunner will copy the local working directory with the project to the server to this path. Note that each experiment produces a separate copy
 (e.g. the above code example would produce 550 copies). Therefore **never** store large files (e.g. datasets) in the local working directory, unless you really have to. In such a case, be sure to exlude them from being copied.
 This can be done by extending the list in the `exlude` parameter of the `create_experiments_helper` function which is called at the bottom of each specification file.  
- <your_binds> - the binds to your singularity. For example, if you want to bind a folder with datasets (recommended) then you will need to bind it: <your-data-path>:<your-data-path>, e.g: /datasets:/datasets.  
- <your_data_path> - path to the datasets on your server
- <path_to_container> - path to the container (should end with `*.sif` extension).

If you want to use wandb:
- <your_api_key> - your WANDB_API_KEY 
- <your_entity_name> - your WANDB entity name (e.g. account name)
- <your_project_name> - your WANDB project name 

Add any other options for singularity you might need in the `singularity_container` line.

## Only conda, no singularity

You can also run mrunner with conda, without singularity. To do so, simply remove the singularity container line from the context specification, and add "conda:<env_name>":
```
contexts:
  <your_context_name>:
    account: <your_slurm_account_name>
    partition: <your_slurm_partition_name>
    backend_type: slurm
    cmd_type: sbatch
    cpu: 5
    mem: 8G
    gpu: 1
    nodes: 1
    ntasks: 1
    slurm_url: <your_server_address>
    storage_dir: <path_on_your_server_where_the_code_will_be_copied>
    conda: <env_name>
```

where <env_name> is the name of your environment. 

## Main Paper:

### Figure 1:
 - `cfashion_mnist/900a_dense.py`
 - `cfashion_mnist/900b_static.py`
 - `cfashion_mnist/900c_dst.py`
 - `mlp_higgs/030a_dense.py`
 - `mlp_higgs/030b_static.py`
 - `mlp_higgs/030c_dst.py`
 - `mlp/001a_dense.py` 
 - `mlp/001b_static.py`
 - `mlp/001c_dst.py`
 - `resnet56/003a_dense.py`
 - `resnet56/003b_static.py`
 - `resnet56/003c_dst.py` 
 - `resnet56-cifar100/910a_dense.py`
 - `resnet56-cifar100/910b_static.py`
 - `resnet56-cifar100/910c_dst.py`
 - `small_conv/033a_dense.py`
 - `small_conv/033b_static.py`
 - `small_conv/033c_dst.py`
 - `tinyimagenet/760a_dense.py`
 - `tinyimagenet/760b_static.py`
 - `tinyimagenet/760c_dst.py`
 - `vgg-cifar100/500a_dense.py`
 - `vgg-cifar100/500b_static.py`
 - `vgg-cifar100/500c_dst.py` 

 
### Figure 2:

Use the scipts in `scripts/ImageNet/` directory for the ImageNet models. Note that those do not work with mrunner, and need to be
manually run on the server. To run the dense model use `./resnet50_desne  <mastert_port_id> <seed>`. Example:
 
```
sbatch specs/scripts/ImageNet/resnet50_dense 10001 1
```

Run the static experiments (`resnet50_static`) analogously. To run the DST methods pass the pruning method name as second argument:

```
sbatch specs/scripts/ImageNet/resnet50_dense 10001 magnitude 1 
```

### Figure 4:
 
 - `mlp/002_update_frequency.py`
 - `resnet56/004_update_frequency.py`
 - `small_conv/034a_update_frequency.py`
 
### Figure 5:
 - `mlp/009a_sparsity_regularizer.py`
 - `mlp/025c_dropout.py`

### Figure 6a:
 - `mlp/035a_jaccard_beg.py`
 - `resnet56/036a_jaccard_beg.py`
 - `small_conv/027a_jaccard_beg.py`

### Figure 6b:
 - `mlp/028c_jaccard.py`
 - `resnet56/030c_jaccard.py`
 - `small_conv/031c_jaccard.py` 
 

### Appendix (selected)

### Figure 7:
 - `mlp/001o_dst_MEST.py`

### Figure 8:
 - `mlp/001n_dst_sensitivity.py`
 
### Figure 11:
 -  `cfashion_mnist/900a_dense_large_batch.py`
 -  `cfashion_mnist/900b_static_large_batch.py`
 -  `cfashion_mnist/900c_dense_large_batch.py`

### Figure 13:
 -  `mlp/006h_global.py`



