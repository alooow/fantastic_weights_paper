import logging
import copy
import os
import hashlib
import wandb
import numpy as np
from collections import defaultdict

logger = None


def setup_logger(args):
    global logger
    if logger is None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    name = '{0}_{1}_{2}.log'.format(args.model, args.density,
                                    hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])
    log_path = os.path.join(args.log_dir, name)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return get_wandb_logger(args)


def info_beginning(args, model):
    print_and_log(model)
    print_and_log('=' * 60)
    print_and_log(args.model)
    print_and_log('=' * 60)
    print_and_log('=' * 60)
    print_and_log('Prune mode: {0}'.format(args.death))
    print_and_log('Growth mode: {0}'.format(args.growth))
    print_and_log('Redistribution mode: {0}'.format(args.redistribution))
    print_and_log('=' * 60)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def get_wandb_logger(args):
    Logger = WandbLogger(args.log_interval, args.save_locally, args.use_wandb, vars(args), save_dir=args.save_dir)
    return Logger


class WandbLogger:
    def __init__(self, log_every=1, save_locally=True, log_to_wandb=False, config=None, save_dir="./save"):
        self.log_every = log_every
        self.step = 0
        self.save_locally = save_locally
        self.data = defaultdict(list)
        self.log_to_wandb = log_to_wandb
        self.log_every = 1
        self.save_dir = save_dir
        if self.log_to_wandb:
            self.init_wandb(config)
        if save_locally:
            os.makedirs(self.save_dir, exist_ok=True)

    def init_wandb(self, config):
        config["cwd"] = os.getcwd()
        if config["distributed"]:
            group = "DDP"
            wandb.require("service")
        else:
            group = None
        wandb.init(project=os.environ["WANDB_PROJECT_NAME"], entity=os.environ["WANDB_ENTITY_NAME"], config=config, tags=[config["tag"]],
                   group=group)

    def log(self, data):
        if (self.step + 1) % self.log_every == 0 or self.step == 0:
            step = self.step
            if self.save_locally:
                for key in data:
                    self.data[key].append(data[key])
                self.data["step"].append(step)

            if self.log_to_wandb:
                wandb.log(data, step=step)
        self.bump()

    def log_no_step(self, key, data):
        if self.save_locally:
            self.data[key].append(data)
        if self.log_to_wandb:
            wandb.log({key:data})

    def bump(self):
        self.step += 1

    def save(self, suffix="results.npy"):
        filename = os.path.join(self.save_dir, suffix)
        np.save(filename, self.data)
        self.reset()

    def reset(self):
        self.data = defaultdict(list)
        self.step = 0

    def finish(self):
        wandb.finish()
