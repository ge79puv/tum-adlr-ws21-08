import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanFromArmStateImage(nn.Module):
    def __init__(self, input_size, n_points, dof, activation=nn.ReLU()):
        super().__init__()
        self.worldsNet = CNN()
        self.downsample = nn.MaxPool2d(4)

        self.mlp1 = MultiLayerPerceptron((256+input_size, 128, 256), 2, activation)
        self.norm1 = nn.BatchNorm1d(256)
        self.highway_layers = Highway(256, 10, activation)
        self.mlp2 = MultiLayerPerceptron((256, 256, 256), 3, activation)
        self.norm2 = nn.BatchNorm1d(256)

        self.activation = activation
        self.fcn64 = nn.Linear(256, n_points*dof)
        self.tanh = nn.Tanh()

    def forward(self, worlds, x):
        
        # worlds_representation = self.worldsNet(worlds)  # image model

        worlds_representation = self.downsample(worlds)  # flattened model
        worlds_representation = torch.flatten(worlds_representation)
        worlds_representation = worlds_representation.reshape(worlds.shape[0], -1)
        # worlds_representation: (worlds_batch_train, 256)

        # x: torch.Size([points_batch_train * worlds_batch_train, 2 * 3])
        worlds_representation = worlds_representation.repeat_interleave(np.int(x.shape[0] / worlds.shape[0]), 0)
        x = torch.cat((worlds_representation, x), 1)  # Merge two branches: batch*(256+2*dof)  torch.Size([50, 262])

        x = self.mlp1(x)
        x = self.norm1(x)
        x = self.highway_layers(x)
        x = self.mlp2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.fcn64(x)
        x = self.tanh(x)
        return x


class Highway(nn.Module):
    def __init__(self, size, num_layers, activation=nn.ReLU()):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.norm = nn.ModuleList([nn.BatchNorm1d(size) for _ in range(num_layers)])
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = self.sigmoid(self.gate[layer](x))
            nonlinear = self.activation(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
            x = self.norm[layer](x)
        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, mpl_size, num_layers, activation=nn.ReLU()):
        super().__init__()
        input_size, hidden_size, output_size = mpl_size
        self.num_layers = num_layers

        self.first = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.last = nn.Linear(hidden_size, output_size)

        self.hidden = None
        if num_layers > 2:
            self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])

    def forward(self, x):
        x = self.first(x)
        x = self.activation(x)

        if self.hidden is not None:
            for layer in range(self.num_layers - 2):
                x = self.hidden[layer](x)
                x = self.activation(x)

        x = self.last(x)
        return x


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(3380, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 3380)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = F.dropout(x)
        x = self.fc4(x)
        return x

