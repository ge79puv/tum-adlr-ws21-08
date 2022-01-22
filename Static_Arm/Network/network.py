import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chompy.GridWorld import create_rectangle_image
from Network.helper import Processing
from Network.points_dataset import StartEndPointsDataset
from chompy.Kinematic.Robots import SingleSphere02
from chompy.parameter import initialize_oc, Parameter


class Dummy(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, output_size)
        self.activation = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(self.norm1(x))

        x = self.linear2(x)
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
        self.num_layers = num_layers
        input_size, hidden_size, output_size = mpl_size
        self.first = nn.Linear(input_size, hidden_size)
        self.hidden = None
        if num_layers > 2:
            self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
            self.norm_hidden = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_layers - 2)])
        self.last = nn.Linear(hidden_size, output_size)
        self.activation = activation
        self.norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.first(x)
        # x = self.norm(x)
        x = self.activation(x)
        if self.hidden is not None:
            for layer in range(self.num_layers - 2):
                x = self.hidden[layer](x)
                # x = self.norm_hidden[layer](x)
                x = self.activation(x)
        x = self.last(x)

        return x


class PredictControlPoints(nn.Module):
    def __init__(self, size, n_layers, n_ctrlpts, activation):
        super().__init__()
        self.fcn64 = nn.ModuleList([MultiLayerPerceptron(size, n_layers, activation)
                                    for _ in range(n_ctrlpts)])
        self.n_ctrpts = n_ctrlpts

    def forward(self, x):
        ctrlpts, weights = None, None
        for i in range(self.n_ctrpts):
            ctrlptsw = self.fcn64[i](x)
            if ctrlpts is None:
                ctrlpts = ctrlptsw[:, 1:][:, None, :]
                weights = ctrlptsw[:, 0][:, None]
            else:
                ctrlpts = torch.cat((ctrlpts, ctrlptsw[:, 1:][:, None, :]), 1)  # (N, n_ctrlpts, dof)
                weights = torch.cat((weights, ctrlptsw[:, 0][:, None]), 1)  # (N, n_ctrlpts,)
        return ctrlpts, weights


class PlanFromState(nn.Module):
    def __init__(self, input_size, n_points, dof, activation=nn.ReLU(), ctrlpts=False):
        super().__init__()
        self.mlp1 = MultiLayerPerceptron((input_size, 128, 256), 2, activation)
        self.norm1 = nn.BatchNorm1d(256)
        # self.highway_layers = Highway(256, 10, activation)
        self.mlp2 = MultiLayerPerceptron((256, 256, 256), 3, activation)
        self.norm2 = nn.BatchNorm1d(256)
        self.activation = activation
        self.ctrlpts = ctrlpts
        if self.ctrlpts:
            self.wp = PredictControlPoints((256, 64, dof + 1), 3, n_points, activation)
        else:
            self.mlp3 = MultiLayerPerceptron((256, 256, n_points*dof), 3, activation)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.norm1(x)
        # x = self.highway_layers(x)
        x = self.mlp2(x)
        x = self.norm2(x)
        x = self.activation(x)
        if self.ctrlpts:
            ctrlpts, weights = self.wp(x)
            result = (ctrlpts, weights)
        else:
            result = self.mlp3(x)
        return result


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        # batch * 1 * 64 * 64
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   # 64-5+1=60
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 30-5+1=26
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(3380, 2048)        # dense layer
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)  # dense layer
        self.fc4 = nn.Linear(512, 256)
        # self.mlp = MultiLayerPerceptron((3380, 256, 256), 4, activation=nn.ReLU())

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))   # 60/2=30
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))   # 26/2=13
        # torch.Size([1, 20, 13, 13])

        x = x.view(-1, 3380)   # # batch*20*13*13 -> batch*3380
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = F.dropout(x)
        x = self.fc4(x)   # batch*256
        # x = self.mlp(x)
        return x


class PlanFromArmState(nn.Module):
    def __init__(self, input_size, n_points, dof, activation=nn.ReLU()):
        super().__init__()
        # self.worldsNet = CNNWorlds(world_voxel=(64,64), activation=nn.ReLU())
        self.mlp1 = MultiLayerPerceptron((input_size, 128, 256), 2, activation)
        self.norm1 = nn.BatchNorm1d(256)
        self.highway_layers = Highway(256, 10, activation)
        self.mlp2 = MultiLayerPerceptron((256, 256, 256), 3, activation)
        self.norm2 = nn.BatchNorm1d(256)
        self.activation = activation
        self.fcn64 = nn.Linear(256, n_points*dof)  # 512
        self.tanh = nn.Tanh()

    def forward(self, worlds, x):
        # x1 = self.worldsNet(worlds)
        x = self.mlp1(x)
        x = self.norm1(x)
        x = self.highway_layers(x)
        x = self.mlp2(x)
        x = self.norm2(x)
        x = self.activation(x)

        # x1 = x1.repeat(np.int(x.shape[0] / worlds.shape[0]), 1)
        # x = torch.cat((x1, x), 1)  # Merge two branches: batch*512  torch.Size([50, 512])

        x = self.fcn64(x)
        x = self.tanh(x)
        return x


class CNNUnit(nn.Module):
    def __init__(self, size, num_layers, kernel_size, activation=nn.ReLU()):
        super().__init__()

        self.num_layers = num_layers
        input_channel, output_channel = size
        self.activation = activation

        self.first = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, (kernel_size, kernel_size), padding=(1, 1)),
            # nn.BatchNorm2d(output_channel),
        )
        self.hidden = nn.ModuleList([nn.Sequential(
            nn.Conv2d(output_channel, output_channel, (kernel_size, kernel_size), padding=(1, 1)),
            # nn.BatchNorm2d(output_channel),
        ) for _ in range(num_layers - 1)])

    def forward(self, x):
        x = self.first(x)
        x = self.activation(x)
        for layer in range(self.num_layers - 1):
            x = self.hidden[layer](x)
            x = self.activation(x)
        return x


class CNNWorlds(nn.Module):
    def __init__(self, world_voxel, activation=nn.ReLU()):
        super().__init__()
        # batch * 1 * 64 * 64
        self.input_size = np.int(world_voxel[0] * world_voxel[1] / 4)
        self.cnn1 = CNNUnit((1, 4), num_layers=3, kernel_size=3, activation=activation)
        self.pooling1 = nn.MaxPool2d(4)
        self.cnn2 = CNNUnit((4, 16), num_layers=3, kernel_size=3, activation=activation)
        self.pooling2 = nn.MaxPool2d(2)
        self.mlp = MultiLayerPerceptron((self.input_size, 256, 256), 4, activation=activation)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.pooling1(x)  # size/4
        x = self.cnn2(x)
        x = self.pooling2(x)  # size/2
        x = x.view(-1, self.input_size)  # torch.Size([10, 1024])
        x = self.mlp(x)  # torch.Size([10, 256])
        return x


class PlanFromImage(nn.Module):
    def __init__(self, world_voxel, input_size, n_points, dof, activation=nn.ReLU(), ctrlpts=False):
        super().__init__()
        self.worldsNet = CNNWorlds(world_voxel, activation)
        self.mlp1 = MultiLayerPerceptron((input_size, 128, 256), 2, activation)
        self.norm1 = nn.BatchNorm1d(256)
        # self.highway_layers = Highway(256, 10, activation)
        self.mlp2 = MultiLayerPerceptron((256, 256, 256), 3, activation)
        self.norm2 = nn.BatchNorm1d(256)
        self.activation = activation
        self.ctrlpts = ctrlpts
        if self.ctrlpts:
            self.wp = PredictControlPoints((512, 64, dof + 1), 3, n_points, activation)
        else:
            self.mlp3 = MultiLayerPerceptron((512, 256, n_points*dof), 3, activation)

    def forward(self, worlds, x):
        x1 = self.worldsNet(worlds)  # torch.Size([1, 256])  torch.Size([10, 256])
        # print(x1.shape) torch.Size([10, 256])
        x2 = self.mlp1(x)
        x2 = self.norm1(x2)
        # x2 = self.highway_layers(x2)
        x2 = self.mlp2(x2)
        x2 = self.norm2(x2)
        x2 = self.activation(x2)

        x1 = x1.repeat(np.int(x.shape[0] / worlds.shape[0]), 1)
        x = torch.cat((x1, x2), 1)  # Merge two branches: batch*512  torch.Size([50, 512])
        if self.ctrlpts:
            ctrlpts, weights = self.wp(x)
            result = (ctrlpts, weights)
        else:
            result = self.mlp3(x)
        return result


# radius = 0.3  # Size of the robot [m]
# robot = SingleSphere02(radius=radius)
# par = Parameter(robot=robot, obstacle_img='rectangle')
# Dof = par.robot.n_dof
# world_limits = np.array([[0, 10],  # x [m]
#                          [0, 10]])  # y [m]
# proc = Processing(world_limits)
# par.robot.limits = world_limits
# n_obstacles = 5
# min_max_obstacle_size_voxel = [3, 15]
# n_voxels = (64, 64)
# img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
# initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)
#
# model = CNNWorlds(n_voxels)
# x = model(torch.from_numpy(img[None, None, :, :]).float())
