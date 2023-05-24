from main import parse_args_default
from data_handling.utils import combine_with_defaults
from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

base_config = {
    "batch_size": 1024,
    "test_batch_size": 1024,
    "multiplier": 1,
    "epochs": 100,
    "lr": 0.01,
    "seed": 17,
    "optimizer": "sgd",
    "data": "fashion_mnist",
    "model" : 'mlp_fmnist',
    "l2": 5.0e-4,
    'bench': False,
    'scaled': False,
    'sparse': True,
    'fix': False,
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
        "seed": [1, 2, 3],
        "density": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.06, 0.07, 0.08, 0.09],
        "scaled": [False],
        "death": ["magnitude", "MEST", "SNIP", "SET", "ReciprocalSensitivity"],
        "growth": ["random", "gradient"],
        "update_frequency": [100]
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
        "results"
    ],
    tags=[name],
    base_config=base_config,
    params_grid=params_grid,
)
