import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariModel(nn.Module):
    def __init__(self, action_count, dropout=0.25):
        """
        Initialize the PyTorch AtariModel Module
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(AtariModel, self).__init__()
        self.action_count = action_count

        # convolutional layer 1  (in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        # convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        # convolutional layer 3
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        # max pooling layer
        #self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 26 * 19, 512)
        self.out = nn.Linear(512, action_count)

        #self.dropout = nn.Dropout(0.25)

    def forward(self, img_array):
        batch_size = img_array.size(0)
        # print("batch_size: {}, sequence_length: {}".format(batch_size, sequence_length))

        # convolutional layers
        # x = self.maxpool(F.relu(self.conv1(img_array)))
        # x = self.maxpool(F.relu(self.conv2(x)))
        # x = self.maxpool(F.relu(self.conv3(x)))

        # convolutional layers
        x = F.relu(self.conv1(img_array))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        #print("x.shape after exiting last max pool: {}".format(x.shape)) #play:  x.shape after exiting last max pool: torch.Size([5, 512, 10, 13])
        # flatten
        x = x.view(-1, 64 * 26 * 19)
        #print("x.view shape: {}".format(x.shape))  #play:  x.view shape: torch.Size([5, 66560])

        # fc layers
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.out(x)

        return x
