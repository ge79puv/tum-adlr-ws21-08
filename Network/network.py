import torch
import torch.nn as nn


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

    def forward(self, x):
        x = self.first(x)
        x = self.norm(x)
        x = self.activation(x)
        if self.hidden is not None:
            for layer in range(self.num_layers-2):
                x = self.hidden[layer](x)
                x = self.norm_hidden[layer](x)
                x = self.activation(x)
        x = self.last(x)

        return x


class Backbone2D(nn.Module):
    def __init__(self, input_size, n_ctrlpts, dof, activation=nn.ReLU()):
        super().__init__()
        self.mlp1 = MultiLayerPerceptron((input_size, 128, 256), 2, activation)
        self.norm1 = nn.BatchNorm1d(256)
        self.highway_layers = Highway(256, 10, activation)
        self.mlp2 = MultiLayerPerceptron((256, 256, 256), 3, activation)
        self.norm2 = nn.BatchNorm1d(256)
        self.activation = activation
        self.fcn64 = nn.ModuleList([MultiLayerPerceptron((256, 64, dof + 1), dof + 1, activation)
                                    for _ in range(n_ctrlpts)])

        self.n_ctrpts = n_ctrlpts

    def forward(self, x):
        x = self.mlp1(x)
        x = self.norm1(x)
        x = self.highway_layers(x)
        x = self.mlp2(x)
        x = self.activation(self.norm2(x))

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
