import torch


'''
                REDISTRIBUTION
'''


def momentum_redistribution(masking, name, weight, mask):
    """Calculates momentum redistribution statistics.

    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.

        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.

        weight      The weight of the respective sparse layer.
                    This is a torch parameter.

        mask        The binary mask. 1s indicated active weights.

    Returns:
        Layer Statistic      The unnormalized layer statistics
                    for the layer "name". A higher value indicates
                    that more pruned parameters are redistributed
                    to this layer compared to layers with lower value.
                    The values will be automatically sum-normalized
                    after this step.


    The calculation of redistribution statistics is the first
    step in this sparse learning library.
    """
    grad = masking.get_momentum_for_weight(weight)
    mean_magnitude = torch.abs(grad[mask.bool()]).mean().item()
    return mean_magnitude


def magnitude_redistribution(masking, name, weight, mask):
    mean_magnitude = torch.abs(weight)[mask.bool()].mean().item()
    return mean_magnitude


def nonzero_redistribution(masking, name, weight, mask):
    nonzero = (weight != 0.0).sum().item()
    return nonzero


def no_redistribution(masking, name, weight, mask):
    num_params = masking.baseline_nonzero
    n = weight.numel()
    return n / float(num_params)
