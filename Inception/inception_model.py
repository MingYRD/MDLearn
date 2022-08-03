import torch
import torch.nn as nn

class Inception(nn.Module):

    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # self.out = c1+c2[1]+c3[1]+c4
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.BatchNorm2d(c2[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0], c2[1], kernel_size=(3, 3), padding=1),
            # nn.Conv2d(c2[0], c2[1], kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(c2[1]),
            nn.ReLU(inplace=True)
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.BatchNorm2d(c3[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c3[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[1], c3[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(c3[2]),
            nn.ReLU(inplace=True)
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return torch.cat([p1, p2, p3, p4], dim=1)


class Inception2(nn.Module):

    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception2, self).__init__()
        # self.out = c1+c2[1]+c3[1]+c4
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.BatchNorm2d(c2[0]),
            nn.ReLU(inplace=True),
            # nn.Conv2d(c2[0], c2[1], kernel_size=(3, 3), padding=1),
            nn.Conv2d(c2[0], c2[1], kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(c2[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[1], c2[2], kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(c2[2]),
            nn.ReLU(inplace=True)
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.BatchNorm2d(c3[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[0], c3[1], kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(c3[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[1], c3[2], kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(c3[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[2], c3[3], kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(c3[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[3], c3[4], kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(c3[4]),
            nn.ReLU(inplace=True)
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return torch.cat([p1, p2, p3, p4], dim=1)


class Inception3(nn.Module):

    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception3, self).__init__()
        # self.out = c1+c2[1]+c3[1]+c4
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.BatchNorm2d(c2[0]),
            nn.ReLU(inplace=True),

        )
        self.p2_1 = nn.Sequential(
            nn.Conv2d(c2[0], c2[1], kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(c2[1]),
            nn.ReLU(inplace=True)
        )
        self.p2_2 = nn.Sequential(
            nn.Conv2d(c2[0], c2[2], kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(c2[2]),
            nn.ReLU(inplace=True)
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.BatchNorm2d(c3[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c3[1]),
            nn.ReLU(inplace=True),
        )
        self.p3_1 = nn.Sequential(
            nn.Conv2d(c3[1], c3[2], kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(c3[2]),
            nn.ReLU(inplace=True)
        )
        self.p3_2 = nn.Sequential(
            nn.Conv2d(c3[1], c3[3], kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(c3[3]),
            nn.ReLU(inplace=True)
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p2_1 = self.p2_1(p2)
        p2_2 = self.p2_2(p2)
        p3 = self.p3(x)
        p3_1 = self.p3_1(p3)
        p3_2 = self.p3_2(p3)
        p4 = self.p4(x)
        return torch.cat([p1, p2_1, p2_2, p3_1, p3_2, p4], dim=1)

