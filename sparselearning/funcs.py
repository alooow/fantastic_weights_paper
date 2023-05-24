from sparselearning.pruners import *
from sparselearning.growers import *
from sparselearning.redistributions import *
from sparselearning.sparsity_initializers import *

prune_funcs = {
    'magnitude': magnitude_prune,
    'SET': magnitude_and_negativity_prune,
    'global_magnitude': global_magnitude_prune
}

growth_funcs = {
    'random': random_growth,
    'momentum': momentum_growth,
    'momentum_neuron': momentum_neuron_growth,
    'gradient': gradient_growth_fixed,
    'global_momentum_growth': global_momentum_growth
}

redistribution_funcs = {
    'momentum': momentum_redistribution,
    'nonzero': nonzero_redistribution,
    'magnitude': magnitude_redistribution,
    'none': no_redistribution
}

sparsity_inits = {

    "global_magnitude": global_magnitude_initializer,
    "snip": snip_initializer,
    "grasp": grasp_initializer,
    "GraSP": grasp_initializer,
    "uniform_plus": uniform_plus_initializer,
    "uniform": uniform_initializer,
    "er": ERK_initializer,
    "erk": ERK_initializer

}
