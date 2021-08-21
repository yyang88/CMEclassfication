import torch.nn as nn
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self, num_classes=2, drop_prob=0.5):
        super().__init__()
        # input_size 1*224*224
        self.drop_prob = drop_prob
        self.resize = transforms.Resize(224)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),  # out 96*26*26
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # out:96*26*26
        )
        self.conv2 = nn.Sequential(
            # in 96*26*26
            nn.Conv2d(96, 256, 5, 1, 2),  # out 256*26*26
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # out 256*12*12
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),  # out 384*12*12
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),  # out 384*12*12
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),  # out 384*12*12
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # out 256*5*5
        )
        self.fc = nn.Sequential(
            # in 6400*1
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.fc(x.view(x.shape[0], -1))
        return out
