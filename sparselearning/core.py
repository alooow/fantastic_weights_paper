from __future__ import print_function
import torch.nn as nn
from sparselearning.funcs import *
import numpy as np
from data_handling.utils import str2bool
import copy
from sparselearning.pruners import SNIPPruner, MESTPruner, ReciprocalSensitivityPruner, MagnitudePruner, SETPruner
import os

def add_sparse_args(parser):
    parser.add_argument('--sparse', type=str2bool, default='true',
                        help='Enable sparse mode. Default: true.')
    parser.add_argument('--fix', type=str2bool, default='true',
                        help='Fix sparse connectivity during training. Default: true.')
    parser.add_argument('--sparse_init', type=str, default='erk', help='sparse initialization',
                        choices=["global_magnitude", "snip", "grasp", "uniform_plus", "uniform", "er", "erk"])
    parser.add_argument('--growth', type=str, default='random', choices=['random', 'momentum', 'gradient'],
                        help='Growth mode. Choose from: momentum, random, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', choices=['magnitude',
                                                                           'SET',
                                                                           'threshold',
                                                                           'SNIP',
                                                                           'MEST',
                                                                           'GradStep',
                                                                           'ReciprocalSensitivity',
                                                                           'GraSP',
                                                                           'TaylorF0',
                                                                           'Random',
                                                                           'SNIPPlus',
                                                                           'RunningMagnitude',
                                                                           'RunningSNIP',
                                                                           'SNIPHalf'],
                        help='Death mode / pruning mode.')
    parser.add_argument('--redistribution', type=str, default='none', choices=['none'],
                        help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death_rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N',
                        help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay_schedule', type=str, default='cosine', choices=['cosine', 'linear'],
                        help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--global_pruning', type=str2bool, default='false',
                        help='Use global pruning instead of per-layer')
    parser.add_argument('--normalize', type=str, choices=["input", "output", "layer", "none"], default="none")
    parser.add_argument('--T', type=float, default=200, help="Temperature for GradSP")
    parser.add_argument('--norm_type', type=str, choices=["sum", "max"], default="sum")
    parser.add_argument('--adjust', type=str, choices=["none", "dim0", "dim1", "total_norm"], default="none")
    parser.add_argument('--jaccard', type=str2bool, default="false")
    parser.add_argument('--save_every_jaccard', type=int, default=800)

def _calculate_masked_fan_in_and_fan_out(self, tensor, mask):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    ones_tensor = torch.ones_like(tensor, device=tensor.device)

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None,
                 death_mode=MagnitudePruner(),
                 growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, global_pruning=False, normalize=None, args=False,
                 train_loader=False, norm_type="sum", opt_order="before", adjust="none", distributed=False):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.train_loader = train_loader
        self.args = args
        self.device = torch.device("cpu") if args.no_cuda else torch.device("cuda")
        self.growth_mode = growth_mode
        self.pruner = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay
        self.density_dict = None
        self.opt_order = opt_order
        self.adjust = adjust

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer
        self.distributed = distributed

        # stats
        self.name2zeros = {}
        self.num_remove = {}
        self.name2nonzeros = {}
        self.name2prune_rate = {}
        self.death_rate = death_rate
        self.steps = 0
        self.norm_type = norm_type

        self.global_pruning = global_pruning
        self.normalize=normalize
        self.running_layer_density = None
        self.after_at_least_one_prune = False
        self.grads_at_update = {}
        self.weights_at_update = {}
        self.scores_at_update = {}
        self.masks_at_update = {}
        self.manual_stop = args.manual_stop

        # if fix, then we do not explore the sparse connectivity
        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency

        if self.args.jaccard:
            self.jaccard = self.args.jaccard
            self.save_every_jaccard = self.args.save_every_jaccard
            self.save_dir = self.args.save_dir
            self.gamma = self.args.gamma
        else:
            self.jaccard = False
            self.save_every_jaccard = None
            self.save_dir = None

    def _sparse_weight_init(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data * self.masks[name]

    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0):
        self.density = density
        self.erk_power_scale = erk_power_scale
        sparsity_inits[mode](self)
        self.apply_mask()

        total_size = 0
        sparse_size = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name in self.masks:
                    print(name, 'density:', (weight != 0).sum().item() / weight.numel())
                    total_size += weight.numel()
                    sparse_size += (weight != 0).sum().int().item()
        print('Total Model parameters:', total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(self.density, sparse_size / total_size))


    def step(self):
        if self.opt_order == "before":
            self.before_step()
        elif self.opt_order == "after":
            self.after_step()
        else:
            raise ValueError("Unknown opt order")

    def store_grads_at_update(self):
        if self.manual_stop:
            for name, weight in self.module.named_parameters():
                self.grads_at_update[name] = copy.copy(weight.grad)
                self.weights_at_update[name] = copy.copy(weight.data)
                if name in self.masks:
                    self.masks_at_update[name] = torch.clone(self.masks[name]).detach()

    def at_end_of_epoch(self):
        pass

    def before_step(self):
        self.optimizer.step()
        self.apply_mask()
        self.death_rate_decay.step()
        self.death_rate = self.death_rate_decay.get_dr()
        self.steps += 1
        self.pruner.step(self)

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.store_grads_at_update()
                self.adjust_prune_rate()
                self.truncate_weights()
                self.running_layer_density = self.print_nonzero_counts()

    def after_step(self):
        #if self.distributed:
        #    self.module._sync_params_and_buffers()
        self.death_rate_decay.step()
        self.death_rate = self.death_rate_decay.get_dr()
        self.steps += 1
        self.pruner.step(self)

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.store_grads_at_update()
                self.adjust_prune_rate()
                self.truncate_weights()
                self.running_layer_density = self.print_nonzero_counts()

        if not self.manual_stop:
            self.optimizer.step()
        self.apply_mask()

    def add_module(self, module, density, sparse_init='ER'):
        self.modules.append(module)
        self.module = module
        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(tensor.device)

        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)
        self.density_dict = self.init(mode=sparse_init, density=density)

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                               np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        self.synchronism_masks()
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data * self.masks[name]

    def adjust_prune_rate(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if name not in self.name2prune_rate: self.name2prune_rate[name] = self.death_rate

                self.name2prune_rate[name] = self.death_rate

                # sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                # if sparsity < 0.2:
                #     # determine if matrix is relativly dense but still growing
                #     expected_variance = 1.0/len(list(self.name2variance.keys()))
                #     actual_variance = self.name2variance[name]
                #     expected_vs_actual = expected_variance/actual_variance
                #     if expected_vs_actual < 1.0:
                #         # growing
                #         self.name2prune_rate[name] = min(sparsity, self.name2prune_rate[name])

    def truncate_weights(self):
        self.after_at_least_one_prune = True
        if self.global_pruning:
            self.global_truncate_weights()
        else:
            self.local_truncate_weights()

    def global_truncate_weights(self):
        assert self.global_pruning, "Pruning must be global to use this truncate method"
        self.pruner(self)

        if not self.manual_stop:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()
                    total_regrowth = self.num_remove[name]
                    # growth
                    new_mask = growth_funcs[self.growth_mode](self, name, new_mask, total_regrowth, weight)
                    new_nonzero = new_mask.sum().item()

                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()

        self.apply_mask()

    def local_truncate_weights(self):

        if self.jaccard:
            pruners = {
                "SNIPFixed": SNIPPruner(),
                "MESTFixed": MESTPruner(self.gamma),
                "ReciprocalSensitivityFixed": ReciprocalSensitivityPruner(),
                "magnitude": MagnitudePruner(),
                "SETFixed": SETPruner(),
                "Random": RandomPruner()
            }
            other_masks = {pname : {} for pname in pruners}
            origs = {}

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
                # death

                if self.jaccard:
                    with torch.no_grad():
                        orig_mask = mask.clone().detach()
                        origs[name] = orig_mask
                        for pname in pruners:
                                mask_copy = mask.clone().detach()
                                weight_copy = weight.clone().detach()
                                weight_copy.grad = weight.grad
                                output_mask = pruners[pname](self, mask_copy, weight_copy, name)
                                other_masks[pname][name] = output_mask

                new_mask = self.pruner(self, mask, weight, name)
                self.num_remove[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                self.masks[name][:] = new_mask
        self.pruner.reset()

        if self.jaccard and self.steps % self.save_every_jaccard == 0:
            j_save_state = {"other_masks": other_masks,
                            "origs": origs}
            filename = os.path.join(self.save_dir, "jaccard-in-time-{}.th".format(self.steps))
            torch.save(j_save_state, filename)

        if not self.manual_stop:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()
                    total_regrowth = self.num_remove[name]
                    # growth
                    new_mask = growth_funcs[self.growth_mode](self, name, new_mask, total_regrowth, weight)
                    new_nonzero = new_mask.sum().item()

                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()

        self.apply_mask()

    '''
                UTILITY
    '''

    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1 / (torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        per_weight_density={}
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros,
                                                               num_nonzeros / float(mask.numel()))
                per_weight_density[name] = num_nonzeros / float(mask.numel())
                print(val)

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break

        return per_weight_density

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item()) / float(
                    self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights / ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights

    def synchronism_masks(self):
        if self.distributed:
            for name in self.masks.keys():
                torch.distributed.broadcast(self.masks[name], src=0, async_op=False)
