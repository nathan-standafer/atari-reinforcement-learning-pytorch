import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariActorModel(nn.Module):
    def __init__(self, action_count):
        super(AtariActorModel, self).__init__()
        self.action_count = action_count

        # convolutional layer 1  (in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.bnconv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bnconv2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        # max pooling layer
        #self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 26 * 19, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, action_count)

    def forward(self, img_array):
        batch_size = img_array.size(0)
        # print("batch_size: {}, sequence_length: {}".format(batch_size, sequence_length))

        # convolutional layers
        # x = self.maxpool(F.relu(self.conv1(img_array)))
        # x = self.maxpool(F.relu(self.conv2(x)))
        # x = self.maxpool(F.relu(self.conv3(x)))

        # convolutional layers
        x = F.relu(self.bnconv1(self.conv1(img_array)))
        x = F.relu(self.bnconv2(self.conv2(x)))
        x = F.relu(self.conv3(x))

        #print("x.shape after exiting last max pool: {}".format(x.shape)) #play:  x.shape after exiting last max pool: torch.Size([5, 512, 10, 13])
        # flatten
        x = x.view(-1, 64 * 26 * 19)
        #print("x.view shape: {}".format(x.shape))  #play:  x.view shape: torch.Size([5, 66560])

        # fc layers
        # x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))

        x = torch.softmax(self.out(x), dim=1)

        return x


class AtariCriticModel(nn.Module):
    def __init__(self, action_count):
        super(AtariCriticModel, self).__init__()
        self.action_count = action_count

        # convolutional layer 1  (in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.bnconv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bnconv2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        # max pooling layer
        #self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 26 * 19, 256)
        self.fc2 = nn.Linear(256 + action_count, 256)
        self.bn2 = nn.BatchNorm1d(256)
        #self.out = nn.Linear(256, action_count)
        self.out = nn.Linear(256, 1)

    def forward(self, img_array, actions):
        batch_size = img_array.size(0)
        # print("batch_size: {}, sequence_length: {}".format(batch_size, sequence_length))

        # convolutional layers
        # x = self.maxpool(F.relu(self.conv1(img_array)))
        # x = self.maxpool(F.relu(self.conv2(x)))
        # x = self.maxpool(F.relu(self.conv3(x)))

        # convolutional layers
        x = F.relu(self.bnconv1(self.conv1(img_array)))
        x = F.relu(self.bnconv2(self.conv2(x)))
        x = F.relu(self.conv3(x))

        #print("x.shape after exiting last max pool: {}".format(x.shape)) #play:  x.shape after exiting last max pool: torch.Size([5, 512, 10, 13])
        # flatten
        x = x.view(-1, 64 * 26 * 19)
        #print("x.view shape: {}".format(x.shape))  #play:  x.view shape: torch.Size([5, 66560])

        # fc layers
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # print("shape of x before catting actions: {}".format(x.shape))
        # print("actions: {}".format(actions))
        x = torch.cat((x, actions), dim=1)  # add in actions to the network
        # print("shape of x after catting actions: {}".format(x.shape))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.out(x)

        return x