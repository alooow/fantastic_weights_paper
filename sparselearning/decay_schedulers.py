import torch
import torch.optim as optim


class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']


class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency
        self.death_rate = death_rate
        self.current_rate = death_rate

    def step(self):
        self.steps += 1

    def get_dr(self):
        if self.steps > 0 and self.steps % self.frequency == 0:
            self.current_rate = self.current_rate * self.factor
            return self.current_rate
        else:
            return self.current_rate


class ConstantDecay(object):
    def __init__(self, death_rate):
        self.death_rate = death_rate
        self.steps = 0

    def step(self):
        self.steps += 1

    def get_dr(self):
        return self.death_rate
