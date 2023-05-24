from sparselearning.core import Masking
from sparselearning.decay_schedulers import CosineDecay, LinearDecay, ConstantDecay
from sparselearning.pruners import *

def get_pruner(args, train_loader, device):
    if not args.global_pruning:
        if args.death == "magnitude":
            return MagnitudePruner()
        elif args.death == "SET":
            return MagnitudeAndNegativityPruner()
        elif args.death == "SNIP":
            return SNIPPruner()
        elif args.death == "MEST":
            return MESTPruner(args.gamma)
        elif args.death == "GradStep":
            return GradStepPruner(args.gamma)
        elif args.death == "Sensitivity":
            return SensitivityPruner()
        elif args.death == "ReciprocalSensitivity":
            return ReciprocalSensitivityPruner()
        elif args.death == "GraSP":
            return GraSPPruner(train_loader, device, T=args.T, intv=args.batch_size, num_iters=1)
        elif args.death == "TaylorF0":
            return TaylorF0Pruner()
        elif args.death == "Random":
            return RandomPruner()
        elif args.death == "SensitivityFixed":
            return SensitivityPruner()
        elif args.death == "SNIPPlus":
            return SNIPPlusPruner()
        elif args.death == "RunningMagnitude":
            return MagnitudeRunningPruner(args.gamma)
        elif args.death == "RunningSNIP":
            return SNIPRunningPruner(args.gamma)
        elif args.death == "SNIPHalf":
            return SNIPHalfPruner()
        else:
            raise ValueError("Unknown pruner")
    else:
        if args.death == "magnitude":
            return MagnitudeGlobalPruner()
        elif args.death == "SET":
            return SETGlobalPruner()
        elif args.death == "SNIP":
            return SNIPGlobalPruner()
        elif args.death == "MEST":
            return MESTGlobalPruner(args.gamma)
        elif args.death == "GradStep":
            return GradStepGlobalPruner(args.gamma)
        elif args.death == "Sensitivity":
            return SensitivityGlobalPruner()
        elif args.death == "ReciprocalSensitivity":
            return ReciprocalSensitivityGlobalPruner()
        elif args.death == "GraSP":
            return GraSPGlobalPruner(train_loader, device, T=args.T, intv=args.batch_size, num_iters=1)
        elif args.death == "TaylorF0":
            return TaylorF0GlobalPruner()
        else:
            raise ValueError("Unknown pruner")


def get_decay(args, train_loader):
    if args.decay_schedule == "cosine":
        decay = CosineDecay(args.death_rate, len(train_loader) * (args.epochs * args.multiplier))
    elif args.decay_schedule == "linear":
        decay = LinearDecay(args.death_rate)
    elif args.decay_schedule == "constant":
        decay = ConstantDecay(args.death_rate)
    else:
        raise ValueError("Unknown decay scheduler")
    return decay


def get_mask(args, model, optimizer, train_loader, device, set_mask=False, multiple_gpu=False):
    pruner = get_pruner(args, train_loader, device)
    mask = None
    if args.sparse:
        decay = get_decay(args, train_loader)
        mask = Masking(optimizer, death_rate=args.death_rate, death_mode=pruner, death_rate_decay=decay,
                       growth_mode=args.growth,
                       redistribution_mode=args.redistribution, global_pruning=args.global_pruning,
                       normalize=args.normalize, args=args, train_loader=train_loader, norm_type=args.norm_type,
                       opt_order=args.opt_order, adjust=args.adjust, distributed=multiple_gpu)
        if set_mask:
            model.mask = mask
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density)
    return mask