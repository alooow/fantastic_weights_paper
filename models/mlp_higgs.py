import torch.nn as nn
import torch.nn.functional as F


class MLP_Higgs(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=256, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return self.fc4(x)



if __name__ == '__main__':
    # check how many parameters in the model
    model = MLP_Higgs(hidden_dim=256)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
