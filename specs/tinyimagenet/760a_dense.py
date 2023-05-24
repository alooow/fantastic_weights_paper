from main import parse_args_default
from data_handling.utils import combine_with_defaults
from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

base_config = {
    "batch_size": 128,
    "test_batch_size": 128,
    "multiplier": 1,
    "epochs": 100,
    "lr": 0.1,
    "seed": 17,
    "optimizer": "sgd",
    "data": "tiny_imagenet",
    "model" :  'efficientnet-b0-plain',
    "l2": 1e-4,
    'bench': False,
    'scaled': False,
    'sparse': False,
    'fix': True,
    'sparse_init': 'erk',
    'growth': 'random',
    'death': 'magnitude',
    'redistribution': 'none',
    'death_rate': 0.50,
    'density': 0.05,
    'update_frequency': 100,
    'decay_schedule': 'cosine',
    'use_wandb': True,
    'save_locally': True,
    'tag': name
}

base_config = combine_with_defaults(
    base_config, defaults=vars(parse_args_default([]))
)

params_grid = [
    {
        "seed": [1, 2, 3, 4, 5],
        "lr": [0.01],
        "l2": [1e-3],
        "nesterov": [False],
        "model": ['efficientnet-b0-plain']
    }
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="weight_importance",
    script="python mrun.py",
    python_path="",
    exclude=[
        ".pytest_cache",
        "__pycache__",
        "checkpoints",
        "out",
        "singularity",
        ".vagrant",
        "notebooks",
        "Vagrantfile",
        "results",
        "data",
        "save",
        ".ipynb_checkpoints",
        "logs",
        "wandb"
    ],
    tags=[name],
    base_config=base_config,
    params_grid=params_grid,
)
