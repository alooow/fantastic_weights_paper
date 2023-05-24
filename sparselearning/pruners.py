import torch
import math
import copy
import torch.nn.functional as F
import torch.autograd as autograd

'''
                PRUNE
'''


class Pruner():
    def __call__(self, masking, mask, weight, name):
        pass

    def reset(self):
        return

    def step(self, masking):
        pass


class MagnitudePruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return magnitude_prune(masking, mask, weight, name)


class MagnitudeRunningPruner(Pruner):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.running_score = {}

    def __call__(self, masking, mask, weight, name):
        return running_magnitude_prune(masking, mask, weight, name, self.running_score)

    def step(self, masking):
        for name, weight in masking.module.named_parameters():
            if name in masking.masks:
                score = abs(weight.data) * masking.masks[name]
                if name in self.running_score:
                    self.running_score[name] = self.running_score[name] * self.beta + (1 - self.beta) * score
                else:
                    self.running_score[name] = score

    def reset(self):
        self.running_score = {}


class SNIPRunningPruner(Pruner):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.running_score = {}

    def __call__(self, masking, mask, weight, name):
        return snip_running_pruning(masking, mask, weight, name, self.running_score)

    def step(self, masking):
        for name, weight in masking.module.named_parameters():
            if name not in masking.masks: continue
            score = abs(weight.data) * abs(weight.grad) * masking.masks[name]
            if name in self.running_score:
                self.running_score[name] = self.running_score[name] * self.beta + (1 - self.beta) * score
            else:
                self.running_score[name] = score

    def reset(self):
        self.running_score = {}


class ThresholdPruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return threshold_death(masking, mask, weight, name)


class MagnitudeAndNegativityPruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return magnitude_and_negativity_prune(masking, mask, weight, name)


class TaylorF0Pruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return taylor_FO(masking, mask, weight, name)



class SNIPHalfPruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return snip_half_pruning(masking, mask, weight, name)


class SNIPPlusPruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return snip_plus_pruning(masking, mask, weight, name)


class SNIPPruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return snip_eps_pruning(masking, mask, weight, name)


class SensitivityPruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return sensitivity_pruning(masking, mask, weight, name)


class ReciprocalSensitivityPruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return reciprocal_sensitivity_fixed_pruning(masking, mask, weight, name)


class SETPruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return SET_fixed(masking, mask, weight, name)


class MESTPruner(Pruner):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def __call__(self, masking, mask, weight, name):
        return mest_pruning_fixed(masking, mask, weight, name, self.gamma)


class GradStepPruner(Pruner):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def __call__(self, masking, mask, weight, name):
        return grad_step_pruning_fixed(masking, mask, weight, name)


class RandomPruner(Pruner):
    def __init__(self):
        super().__init__()

    def __call__(self, masking, mask, weight, name):
        return random_pruning(masking, mask, weight, name)


class GraSPPruner(Pruner):
    def __init__(self, train_dataloader, device, T=200, intv=128, num_iters=1):
        super().__init__()
        self.train_loader = train_dataloader
        self.T = T
        self.intv = intv
        self.num_iters = num_iters
        self.device = device
        self.computed = False
        self.scores_names = None

    def reset(self):
        self.computed = False
        self.scores_names = None
        return

    def compute_gradients(self, masking):
        if not self.computed:
            self.scores_names = GraSP(masking, self.train_loader, self.num_iters, self.intv, self.device, self.T)
            self.computed = True

    def __call__(self, masking, mask, weight, name):
        num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
        num_zeros = masking.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        if num_remove == 0.0: return mask

        self.compute_gradients(masking)
        score = self.scores_names[name].clone()
        score = norm_score(masking, score, weight)
        score[(1 - mask).to(bool)] = torch.inf

        x, idx = torch.sort(score.flatten(), descending=True)
        mask.data.view(-1)[idx[:k]] = 0.0
        return mask


def get_norm_value(masking, score, dim=None):
    if masking.norm_type == "max":
        if dim is not None:
            value = torch.amax(score, dim=dim, keepdim=True)
        else:
            value = torch.amax(score)
    elif masking.norm_type == "sum":
        if dim is not None:
            value = torch.sum(score, dim=dim, keepdim=True)
        else:
            value = torch.sum(score)
    else:
        raise ValueError("Unknown normalization type")
    return value


def norm_score(masking, score, weight):
    if masking.normalize is not None:
        with torch.no_grad():
            if masking.normalize == "input":
                assert len(weight.data.shape) == 2, "Implemented only for FC"
                value = get_norm_value(masking, score, 1)
                score /= value
            elif masking.normalize == "output":
                assert len(weight.data.shape) == 2, "Implemented only for FC"
                value = get_norm_value(masking, score, 0)
                score /= value
            elif masking.normalize == "channel":
                if len(weight.data.shape) == 4:
                    value = get_norm_value(masking, score, (1, 2, 3))
                else:
                    print("Not a convolutional layer, falling back to layer-wise normalization")
                    value = get_norm_value(masking, score, None)
                score /= value
            elif masking.normalize == "depth":
                if len(weight.data.shape) == 4:
                    value = get_norm_value(masking, score, (2, 3))
                else:
                    print("Not a convolutional layer, falling back to layer-wise normalization")
                    value = get_norm_value(masking, score, None)
                score /= value
            elif masking.normalize == "layer":
                value = get_norm_value(masking, score, None)
                score /= value
            elif masking.normalize == "none":
                return score
            else:
                raise ValueError("Unknown normalization scheme")
    return score


def magnitude_prune(masking, mask, weight, name):
    """Prunes the weights with smallest magnitude.

    The pruning functions in this sparse learning library
    work by constructing a binary mask variable "mask"
    which prevents gradient flow to weights and also
    sets the weights to zero where the binary mask is 0.
    Thus 1s in the "mask" variable indicate where the sparse
    network has active weights. In this function name
    and masking can be used to access global statistics
    about the specific layer (name) and the sparse network
    as a whole.

    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.

        mask        The binary mask. 1s indicated active weights.

        weight      The weight of the respective sparse layer.
                    This is a torch parameter.

        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.

    Returns:
        mask        Pruned Binary mask where 1s indicated active
                    weights. Can be modified in-place or newly
                    constructed

    Accessable global statistics:

    Layer statistics:
        Non-zero count of layer:
            masking.name2nonzeros[name]
        Zero count of layer:
            masking.name2zeros[name]
        Redistribution proportion:
            masking.name2variance[name]
        Number of items removed through pruning:
            masking.name2removed[name]

    Network statistics:
        Total number of nonzero parameter in the network:
            masking.total_nonzero = 0
        Total number of zero-valued parameter in the network:
            masking.total_zero = 0
        Total number of parameters removed in pruning:
            masking.total_removed = 0
    """
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0

    score = torch.abs(weight.data)
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def running_magnitude_prune(masking, mask, weight, name, running_scores):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0

    score = running_scores[name]
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def rescale(masking, old_mask, new_mask, weight, name):
    with torch.no_grad():
        if masking.adjust == "dim0":
            norm_before = torch.norm(weight.data * old_mask, dim=0, keepdim=True)
            norm_after = torch.norm(weight.data * new_mask, dim=0, keepdim=True)
            ratio = torch.nan_to_num(norm_before / norm_after, nan=0.0)
            weight.data = weight.data * ratio
        elif masking.adjust == "dim1":
            norm_before = torch.norm(weight.data * old_mask, dim=1, keepdim=True)
            norm_after = torch.norm(weight.data * new_mask, dim=1, keepdim=True)
            ratio = torch.nan_to_num(norm_before / norm_after, nan=0.0)
            weight.data = weight.data * ratio
        elif masking.adjust == "total_norm":
            norm_before = torch.norm(weight.data * old_mask)
            norm_after = torch.norm(weight.data * new_mask)
            ratio = torch.nan_to_num(norm_before / norm_after, nan=0.0)
            weight.data = weight.data * ratio
        elif masking.adjust == "none":
            pass
        else:
            raise ValueError("Unknown weight adjustment")


def threshold_death(masking, mask, weight, name):
    return (torch.abs(weight.data) > masking.threshold)


def taylor_FO(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = (weight.data * weight.grad).pow(2) * mask
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    mask.data.view(-1)[idx[:k]] = 0.0

    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)
    return mask


def random_pruning(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = (torch.randperm(len(weight.view(-1))) + 1).to(masking.device) * mask.view(-1)
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score)
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)
    return mask


def mest_pruning(masking, mask, weight, name, gamma=1.0):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = abs(weight.data) + gamma * abs(weight.grad)
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def mest_pruning_fixed(masking, mask, weight, name, gamma=1.0):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = (abs(weight.data) + gamma * abs(weight.grad)) * mask
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def snip_pruning(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = abs(weight.data * weight.grad)
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def snip_eps_pruning(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = (abs(weight.data * weight.grad) + 1e-5) * mask
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def snip_running_pruning(masking, mask, weight, name, running_score):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = running_score[name]
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def snip_half_pruning(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    nz = math.ceil(num_zeros)
    k_half = math.ceil(num_remove // 2)
    left = nz + k_half
    right = k_half

    score = abs(weight.data * weight.grad) * mask
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:left]] = 0.0
    mask.data.view(-1)[idx[-right:]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def snip_plus_pruning(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = (weight.data * weight.grad)
    score = norm_score(masking, score, weight)
    score[(1 - mask).to(bool)] = torch.inf
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten(), descending=True)
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def grad_step_pruning(masking, mask, weight, name, gamma=1.0):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = abs(weight.data - gamma * weight.grad)
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def grad_step_pruning_fixed(masking, mask, weight, name, gamma=1.0):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = abs(weight.data - gamma * weight.grad) * mask
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def sensitivity_pruning(masking, mask, weight, name, epsilon=1e-8):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    score = ((abs(weight.grad) / (abs(weight.data) + epsilon)) + 1) * mask
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def reciprocal_sensitivity_pruning(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = abs(weight.data) / abs(weight.grad)
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def reciprocal_sensitivity_fixed_pruning(masking, mask, weight, name, epsilon=1e-8):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)

    score = ((abs(weight.data) / (abs(weight.grad) + epsilon)) + 1) * mask
    score = norm_score(masking, score, weight)
    if masking.manual_stop:
        masking.scores_at_update[name] = score

    x, idx = torch.sort(score.flatten())
    old_mask = mask.clone()
    mask.data.view(-1)[idx[:k]] = 0.0
    rescale(masking, old_mask, mask, weight, name)

    return mask


def magnitude_and_negativity_prune(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])

    if masking.normalize is not None and masking.normalize != "none":
        raise NotImplementedError("Normalizing not implemented for SET")
    # find magnitude threshold
    # remove all weights which absolute value is smaller than threshold

    score = weight[weight > 0.0].data

    x, idx = torch.sort(score.view(-1))
    k = math.ceil(num_remove / 2.0)
    if k >= x.shape[0]:
        k = x.shape[0]

    threshold_magnitude = x[k - 1].item()

    # find negativity threshold
    # remove all weights which are smaller than threshold

    score = weight[weight < 0.0].data

    x, idx = torch.sort(score.view(-1))
    k = math.ceil(num_remove / 2.0)
    if k >= x.shape[0]:
        k = x.shape[0]
    threshold_negativity = x[k - 1].item()

    pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
    neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)

    new_mask = pos_mask | neg_mask
    rescale(masking, mask, new_mask, weight, name)

    if masking.manual_stop:
        masking.scores_at_update[name] = abs(weight.data)

    return new_mask


def SET_fixed(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])

    if masking.normalize is not None and masking.normalize != "none":
        raise NotImplementedError("Normalizing not implemented for SET")
    # find magnitude threshold
    # remove all weights which absolute value is smaller than threshold
    score = weight[weight > 0.0].data

    x, idx = torch.sort(score.view(-1))
    k = math.ceil(num_remove / 2.0)
    if k >= x.shape[0]:
        k = x.shape[0]

    z = k - 1 if k > 0 else k
    if len(x) > 0:
        threshold_magnitude = x[z].item()
    else:
        threshold_magnitude = 0.0

    # find negativity threshold
    # remove all weights which are smaller than threshold
    score = weight[weight < 0.0].data

    x, idx = torch.sort(score.view(-1), descending=True)
    k = math.ceil(num_remove / 2.0)

    if k >= x.shape[0]:
        k = x.shape[0]
    z = k - 1 if k > 0 else k
    if len(x) > 0:
        threshold_negativity = x[z].item()
    else:
        threshold_negativity = 0.0

    pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
    neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)

    new_mask = pos_mask | neg_mask
    rescale(masking, mask, new_mask, weight, name)

    if masking.manual_stop:
        masking.scores_at_update[name] = abs(weight.data)

    return new_mask


# ----------------------------------------------- GLOBAL PRUNERS ---------------------------------------------------- #


class GlobalPruner():
    def __call__(self, masking):
        score_threshold = self.get_threshold(masking)
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                score = self.score_weight(weight, masking.masks[name])
                score = norm_score(masking, score, weight)
                masking.masks[name] = (score > score_threshold).float()
                masking.num_remove[name] = int(masking.name2nonzeros[name] - masking.masks[name].sum().item())

    def score_weight(self, weight, mask):
        pass

    def get_threshold(self, masking):
        prune_rate = masking.death_rate
        num_zeros = 0.0
        num_nonzeros = 0.0

        scores = []
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                mask = masking.masks[name]
                masking.name2nonzeros[name] = mask.sum().item()
                masking.name2zeros[name] = mask.numel() - masking.name2nonzeros[name]

                num_zeros += masking.name2zeros[name]
                num_nonzeros += masking.name2nonzeros[name]
                score = self.score_weight(weight, mask)
                score = norm_score(masking, score, weight)
                scores.append(score)

        num_remove = math.ceil(prune_rate * num_nonzeros)
        k = math.ceil(num_zeros + num_remove)

        all_scores = torch.cat([x.view(-1) for x in scores])
        x, idx = torch.sort(all_scores)
        return x[k - 1]

    def step(self, masking):
        pass

class GraSPGlobalPruner(GlobalPruner):
    def __init__(self, train_dataloader, device, T=200, intv=128, num_iters=1):
        super().__init__()
        self.train_loader = train_dataloader
        self.T = T
        self.intv = intv
        self.num_iters = num_iters
        self.device = device

    def __call__(self, masking):
        scores = self.compute_scores(masking)
        score_threshold = self.get_GraSP_threshold(masking, scores)
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                score = scores[name]
                score = norm_score(masking, score, weight)
                masking.masks[name] = (score < score_threshold).float()
                masking.num_remove[name] = int(masking.name2nonzeros[name] - masking.masks[name].sum().item())

    def compute_scores(self, masking):
        return GraSP(masking, self.train_loader, self.num_iters, self.intv, self.device, self.T)

    def score_weight(self, weight, mask):
        pass

    def get_threshold(self, masking):
        pass


    def get_GraSP_threshold(self, masking, score_dict):
        prune_rate = masking.death_rate
        num_zeros = 0.0
        num_nonzeros = 0.0
        with torch.no_grad():
            scores = []
            for module in masking.modules:
                for name, weight in module.named_parameters():
                    if name not in masking.masks: continue
                    mask = masking.masks[name]
                    masking.name2nonzeros[name] = mask.sum().item()
                    masking.name2zeros[name] = mask.numel() - masking.name2nonzeros[name]

                    num_zeros += masking.name2zeros[name]
                    num_nonzeros += masking.name2nonzeros[name]
                    score = score_dict[name].clone()
                    score = norm_score(masking, score, weight)
                    score[(1 - mask).to(bool)] = torch.inf
                    scores.append(score)

            num_remove = math.ceil(prune_rate * num_nonzeros)
            k = math.ceil(num_zeros + num_remove)

            all_scores = torch.cat([x.view(-1) for x in scores])
            x, idx = torch.sort(all_scores, descending=True)
            return x[k - 1]


class MagnitudeGlobalPruner(GlobalPruner):
    def __init__(self):
        super().__init__()

    def score_weight(self, weight, mask):
        return abs(weight)


class TaylorF0GlobalPruner(GlobalPruner):
    def __init__(self):
        super().__init__()

    def score_weight(self, weight, mask):
        return (weight.data * weight.grad).pow(2) * mask



class MESTGlobalPruner(GlobalPruner):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def score_weight(self, weight, mask):
        return (abs(weight.data) + self.gamma * abs(weight.grad)) * mask


class SNIPGlobalPruner(GlobalPruner):
    def __init__(self):
        super().__init__()

    def score_weight(self, weight, mask):
        return (abs(weight.data * weight.grad)+1e-8) * mask

class GradStepGlobalPruner(GlobalPruner):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def score_weight(self, weight, mask):
        return (abs(weight.data - self.gamma * weight.grad)) * mask


class SensitivityGlobalPruner(GlobalPruner):
    def __init__(self):
        super().__init__()

    def score_weight(self, weight, mask):
        return abs(weight.grad) / (abs(weight.data) + 1e-8)


class ReciprocalSensitivityGlobalPruner(GlobalPruner):
    def __init__(self):
        super().__init__()

    def score_weight(self, weight, mask):
        return abs(weight.data) / (abs(weight.grad) + 1e-8)


class SETLikeGlobalPruner(GlobalPruner):
    def __init__(self, descending):
        super().__init__()
        self.descending = descending

    def __call__(self, masking):
        if masking.normalize is not None and masking.normalize != "none":
            raise NotImplementedError("Normalizing not implemented for SET")
        threshold_magnitude, threshold_negativity = self.get_threshold(masking)
        self.threshold_magnitude = threshold_magnitude
        self.threshold_negativity = threshold_negativity
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                masking.masks[name] = self.get_mask(weight).to(masking.device)
                masking.num_remove[name] = int(masking.name2nonzeros[name] - masking.masks[name].sum().item())

    def score_weight(self, weight, mask):
        raise NotImplementedError()

    def get_mask(self, weight):
        return ((weight.data > self.threshold_magnitude) & (weight.data > 0.0)) | (
                (weight.data < self.threshold_negativity) & (weight.data < 0.0))

    def get_threshold(self, masking):
        prune_rate = masking.death_rate
        num_zeros = 0.0
        num_nonzeros = 0.0

        weight_pos = []
        weight_neg = []
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                mask = masking.masks[name]
                masking.name2nonzeros[name] = mask.sum().item()
                masking.name2zeros[name] = mask.numel() - masking.name2nonzeros[name]

                num_zeros += masking.name2zeros[name]
                num_nonzeros += masking.name2nonzeros[name]
                weight_pos.append(weight[weight > 0.0].view(-1))
                weight_neg.append(weight[weight < 0.0].view(-1))

        num_remove = math.ceil(prune_rate * num_nonzeros)
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort(torch.cat(weight_pos))
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        z = k - 1 if k > 0 else k
        if len(x) > 0:
            threshold_magnitude = x[z].item()
        else:
            threshold_magnitude = 0

            # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(torch.cat(weight_neg), descending=self.descending)
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        z = k - 1 if k > 0 else k
        if len(x) > 0:
            threshold_negativity = x[z].item()
        else:
            threshold_negativity = 0

        return threshold_magnitude, threshold_negativity


class MagnitudeAndNegativityGlobalPruner(SETLikeGlobalPruner):
    def __init__(self):
        super().__init__(False)


class SETGlobalPruner(SETLikeGlobalPruner):
    def __init__(self):
        super().__init__(True)


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


def global_magnitude_prune(masking):
    prune_rate = 0.0
    for name in masking.name2prune_rate:
        if name in masking.masks:
            prune_rate = masking.name2prune_rate[name]
    tokill = math.ceil(prune_rate * masking.baseline_nonzero)
    total_removed = 0
    prev_removed = 0
    while total_removed < tokill * (1.0 - masking.tolerance) or (total_removed > tokill * (1.0 + masking.tolerance)):
        total_removed = 0
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                remain = (torch.abs(weight.data) > masking.prune_threshold).sum().item()
                total_removed += masking.name2nonzeros[name] - remain

        if prev_removed == total_removed: break
        prev_removed = total_removed
        if total_removed > tokill * (1.0 + masking.tolerance):
            masking.prune_threshold *= 1.0 - masking.increment
            masking.increment *= 0.99
        elif total_removed < tokill * (1.0 - masking.tolerance):
            masking.prune_threshold *= 1.0 + masking.increment
            masking.increment *= 0.99

    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks: continue
            masking.masks[name][:] = torch.abs(weight.data) > masking.prune_threshold

    return int(total_removed)


def GraSP(masking, train_loader, num_iters, intv, device, T):
    net = copy.deepcopy(masking.module)
    net.zero_grad()
    weights = []
    for name, weight in net.named_parameters():
        if name in masking.masks:
            weights.append(weight)

    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    dataloader_iter = iter(train_loader)
    for it in range(num_iters):
        inputs, targets = next(dataloader_iter)
        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)

        start = 0

        while start < N:
            end = min(start + intv, N)
            # print('(1):  %d -> %d.' % (start, end))
            inputs_one.append(din[start:end])
            targets_one.append(dtarget[start:end])
            outputs = net.forward(inputs[start:end].to(device)) / T  # divide by temperature to make it uniform
            loss = F.cross_entropy(outputs, targets[start:end].to(device))
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]
            start = end

    for it in range(len(inputs_one)):
        # print("(2): Iterations %d/%d." % (it, len(inputs_one)))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        outputs = net.forward(inputs) / T  # divide by temperature to make it uniform
        loss = F.cross_entropy(outputs, targets)
        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for name, weight in net.named_parameters():
            if name in masking.masks:
                z += (grad_w[count] * grad_f[count]).sum()
                count += 1
        z.backward()

    grads_dict = dict()
    for name, weight in net.named_parameters():
        if name in masking.masks:
            grads_dict[name] = (-weight.data * weight.grad).detach().to(device)

    return grads_dict
