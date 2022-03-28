import torch.nn as nn

class LogisticRegressor(nn.Module):

    def __init__(self, input_size=784, num_classes=10):
        super(LogisticRegressor, self).__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, num_classes, bias=True)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        logits = self.linear(x)
        return logits

class Logistic2NN(nn.Module):

    def __init__(self, input_size=784, num_classes=10):
        super(Logistic2NN, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, 100, bias=True)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(100, num_classes, bias=True)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.linear1(x)
        out = self.act1(out)
        logits = self.linear2(out)
        return logits

