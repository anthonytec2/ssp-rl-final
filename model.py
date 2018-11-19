import torch.nn as nn
import torch.nn.functional as F


class pg_model(nn.Module):
    def __init__(self):
        super(pg_model, self).__init__()
        self.l1 = nn.Linear(4, 10)
        self.l2 = nn.Linear(10, 2)
        self.l3 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.softmax(x, dim=1)
        return x
