import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, size, num_layers, activation=nn.ReLU()):

        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.activation = activation

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.activation(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
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
        self.last = nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.first(x))
        if self.hidden is not None:
            for layer in range(self.num_layers-2):
                x = self.activation(self.hidden[layer](x))
        x = self.activation(self.last(x))
        return x


class Backbone2D(nn.Module):
    def __init__(self, input_size, output_size, activation=nn.ReLU()):
        super().__init__()
        self.mlp1 = MultiLayerPerceptron((input_size, 128, 256), 2, activation)
        self.highway_layers = Highway(256, 10, activation)
        self.mlp2 = MultiLayerPerceptron((256, 256, 256), 3, activation)
        self.fcn64 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.highway_layers(x)
        x = self.mlp2(x)
        x = self.fcn64(x)
        return x


class Dummy(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x
