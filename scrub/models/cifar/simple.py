from torch import nn
import torch.nn.functional as F
import torch

class CIFAR10Logistic2NN(nn.Module):

    def __init__(self, input_size=32*32*3, num_classes=10):
        super(CIFAR10Logistic2NN, self).__init__()
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

class Net(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            #nn.MaxPool2d(2),
            #nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            #nn.MaxPool2d(2),
            #nn.ReLU(),
            nn.Conv2d(20, 30, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(30, 40, kernel_size=3),
            #nn.MaxPool2d(2),
            #nn.ReLU(),
            #nn.Conv2d(40, 50, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(640, 50), # 64x64
            #nn.Linear(26450, n_classes), # 224x224
            #nn.ReLU(),
            #nn.Dropout(),
            #nn.Linear(n_classes, n_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        #print(features.shape)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        #self.conv3 = nn.Conv2d(20, 30, 5)
        self.fc1 = nn.Linear(500, 250)
        self.fc2 = nn.Linear(250, 100)
        #self.fc3 = nn.Linear(600, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CIFAR10_CNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CIFAR10_CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x
