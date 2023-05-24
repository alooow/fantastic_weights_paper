import torch
from sparselearning.snip import SNIP, GraSP
import numpy as np


def global_magnitude_initializer(masking):
    print('initialize by global magnitude')
    weight_abs = []
    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks: continue
            weight_abs.append(torch.abs(weight))
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
    num_params_to_keep = int(len(all_scores) * masking.density)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]
    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks: continue
            masking.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()


def grasp_initializer(masking):
    layer_wise_sparsities = GraSP(masking.module, masking.density, masking.train_loader, masking.device)
    # re-sample mask positions
    for sparsity_, name in zip(layer_wise_sparsities, masking.masks):
        masking.masks[name][:] = (torch.rand(masking.masks[name].shape) < (1 - sparsity_)).float().data.to(masking.device)


def snip_initializer(masking):
    layer_wise_sparsities = SNIP(masking.module, masking.density, masking.train_loader, masking.device)
    # re-sample mask positions
    for sparsity_, name in zip(layer_wise_sparsities, masking.masks):
        masking.masks[name][:] = (torch.rand(masking.masks[name].shape) < (1 - sparsity_)).float().data.to(masking.device)

def ERK_initializer(masking):
    print('initialize by ERK')
    total_params = 0
    for name, weight in masking.masks.items():
        total_params += weight.numel()
        if 'classifier' in name:
            masking.fc_params = weight.numel()
    is_epsilon_valid = False
    dense_layers = set()
    while not is_epsilon_valid:

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, mask in masking.masks.items():
            n_param = np.prod(mask.shape)
            n_zeros = n_param * (1 - masking.density)
            n_ones = n_param * masking.density

            if name in dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros

            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones
                # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                raw_probabilities[name] = (
                                                  np.sum(mask.shape) / np.prod(mask.shape)
                                          ) ** masking.erk_power_scale
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    print(f"Sparsity of var:{mask_name} had to be set to 0.")
                    dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    density_dict = {}
    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, mask in masking.masks.items():
        n_param = np.prod(mask.shape)
        if name in dense_layers:
            density_dict[name] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[name]
            density_dict[name] = probability_one
        print(
            f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
        )
        masking.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.to(masking.device)

        total_nonzero += density_dict[name] * mask.numel()
    print(f"Overall sparsity {total_nonzero / total_params}")
    return density_dict


def uniform_initializer(masking):
    masking.baseline_nonzero = 0
    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks: continue
            masking.masks[name][:] = (torch.rand(weight.shape) < masking.density).float().data.to(masking.device)
            masking.baseline_nonzero += weight.numel() * masking.density


def uniform_plus_initializer(masking):
    total_params = 0
    for name, weight in masking.masks.items():
        total_params += weight.numel()
    total_sparse_params = total_params * masking.density
    # remove the first layer
    total_sparse_params = total_sparse_params - masking.masks['conv.weight'].numel()
    masking.masks.pop('conv.weight')
    if masking.density < 0.2:
        total_sparse_params = total_sparse_params - masking.masks['fc.weight'].numel() * 0.2
        masking.density = float(total_sparse_params / total_params)

        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                if name != 'fc.weight':
                    masking.masks[name][:] = (torch.rand(weight.shape) < masking.density).float().data.to(masking.device)
                else:
                    masking.masks[name][:] = (torch.rand(weight.shape) < 0.2).float().data.to(masking.device)
    else:
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                masking.masks[name][:] = (torch.rand(weight.shape) < masking.density).float().data.to(masking.device)


