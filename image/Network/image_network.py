import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-2)])
            self.norm_hidden = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_layers-2)])
        self.last = nn.Linear(hidden_size, output_size)
        self.activation = activation
        self.norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.first(x)
        x = self.norm(x)
        x = self.activation(x)
        # x = self.dropout(x)
        if self.hidden is not None:
            for layer in range(self.num_layers-2):
                x = self.hidden[layer](x)
                x = self.norm_hidden[layer](x)
                x = self.activation(x)
        x = self.last(x)

        return x


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        # batch * 1 * 64 * 64
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   # 64-5+1=60
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 30-5+1=26
        self.conv2_drop = nn.Dropout2d()        # No ResNet-50, black-white image
        self.fc1 = nn.Linear(3380, 1024)        # dense layer
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))   # 60/2=30
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))   # 26/2=13
        x = x.view(-1, 3380)   # # batch*20*13*13 -> batch*3380
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return x


class Backbone2D(nn.Module):
    def __init__(self, input_size, n_ctrlpts, dof, activation=nn.ReLU()):
        super().__init__()
        self.cnn = CNN()
        self.mlp1 = MultiLayerPerceptron((input_size, 128, 256), 2, activation)
        self.norm1 = nn.BatchNorm1d(256)
        self.highway_layers = Highway(256, 10, activation)
        self.mlp2 = MultiLayerPerceptron((256, 256, 256), 3, activation)
        self.norm2 = nn.BatchNorm1d(256)
        self.activation = activation
        self.fcn64 = nn.ModuleList([MultiLayerPerceptron((512, 64, dof + 1), dof + 1, activation)
                                    for _ in range(n_ctrlpts)])

        self.n_ctrpts = n_ctrlpts

    def forward(self, x, world):
        x1 = self.cnn(world)   # torch.Size([1, 256])  torch.Size([10, 256])
        if x1.shape[0] == 1:
            x1 = x1.expand(x.shape[0], x1.shape[1])  # # torch.Size([50, 256])  50张一样的image，50对起始点
        elif x1.shape[0] == 5:
            x1 = x1.repeat(10, 1)
        x2 = self.mlp1(x)
        x2 = self.norm1(x2)
        x2 = self.highway_layers(x2)
        x2 = self.mlp2(x2)
        x2 = self.activation(self.norm2(x2))   # batch*256  torch.Size([50, 256])
        x = torch.cat((x1, x2), 1)  # Merge: batch*512  torch.Size([50, 512])

        ctrlpts, weights = None, None
        for i in range(self.n_ctrpts):
            ctrlptsw = self.fcn64[i](x)  # torch.Size([50, 3])
            if ctrlpts is None:
                ctrlpts = ctrlptsw[:, 1:][:, None, :]
                weights = ctrlptsw[:, 0][:, None]
            else:
                ctrlpts = torch.cat((ctrlpts, ctrlptsw[:, 1:][:, None, :]), 1)  # (N, n_ctrlpts, dof)
                weights = torch.cat((weights, ctrlptsw[:, 0][:, None]), 1)  # (N, n_ctrlpts,)
        # print(weights)
        # print(ctrlpts.shape)
        return ctrlpts, weights

