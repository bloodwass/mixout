import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.linear1 = nn.Linear(784, 300)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)
        self.linear3 = nn.Linear(100, 10)
    
    def forward(self, input):
        input = self.drop1(self.relu1(self.linear1(input)))
        input = self.drop2(self.relu2(self.linear2(input)))
        return self.linear3(input)