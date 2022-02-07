import numpy as np
import torch
import torch.nn as nn


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
            self.mlp3 = MultiLayerPerceptron((256, 256, n_points*dof), 2, activation)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.norm1(x)
        # x = self.highway_layers(x)
        x = self.activation(x)
        x = self.mlp2(x)
        x = self.norm2(x)
        x = self.activation(x)
        if self.ctrlpts:
            ctrlpts, weights = self.wp(x)
            result = (ctrlpts, weights)
        else:
            result = self.mlp3(x)
        return result


class PlanFromImage(nn.Module):
    def __init__(self, world_voxel, input_size, n_points, dof, activation=nn.ReLU(), ctrlpts=False, encoder=None):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.BatchNorm1d(32*32),
            MultiLayerPerceptron((32*32, 256, 64), 2, nn.ReLU()),
            nn.BatchNorm1d(64),
            # nn.ReLU(),
            # MultiLayerPerceptron((256, 128, 64), 3, nn.ReLU())
        )
        if encoder:
            self.encoder.load_state_dict(encoder)

        self.mlp1 = MultiLayerPerceptron((64+input_size, 128, 256), 2, activation)
        self.norm1 = nn.BatchNorm1d(256)
        # self.highway_layers = Highway(128, 10, activation)
        self.mlp2 = MultiLayerPerceptron((256, 256, 256), 3, activation)
        self.norm2 = nn.BatchNorm1d(256)
        self.activation = activation

        self.ctrlpts = ctrlpts
        if self.ctrlpts:
            self.wp = PredictControlPoints((256, 64, dof + 1), 3, n_points, activation)
        else:
            self.mlp3 = MultiLayerPerceptron((256, 256, n_points*dof), 2, activation)

    def forward(self, worlds, x):
        x1 = self.encoder(worlds)
        x1 = x1.repeat_interleave(np.int(x.shape[0] / worlds.shape[0]), dim=0)
        x2 = torch.cat((x1, x), 1)

        x2 = self.mlp1(x2)
        x2 = self.norm1(x2)
        x3 = self.activation(x2)
        # x3 = self.highway_layers(x3)
        x3 = self.mlp2(x3)
        x3 = self.norm2(x3)
        x3 = self.activation(x3)

        if self.ctrlpts:
            ctrlpts, weights = self.wp(x3)
            result = (ctrlpts, weights)
        else:
            result = self.mlp3(x3)
        return result


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
        x = self.norm(x)
        x = self.activation(x)
        if self.hidden is not None:
            for layer in range(self.num_layers - 2):
                x = self.hidden[layer](x)
                x = self.norm_hidden[layer](x)
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


class CNNUnit(nn.Module):
    def __init__(self, size, num_layers, kernel_size, stride=(1, 1), activation=nn.ReLU()):
        super().__init__()

        self.num_layers = num_layers
        input_channel, output_channel = size
        self.activation = activation

        self.first = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, (kernel_size, kernel_size),
                      stride, padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(output_channel),
        )
        self.hidden = nn.ModuleList([nn.Sequential(
            nn.Conv2d(output_channel, output_channel, (kernel_size, kernel_size),
                      padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(output_channel),
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
        self.cnn1 = CNNUnit((1, 4), num_layers=2, kernel_size=3, stride=2, activation=activation)
        self.pooling1 = nn.MaxPool2d(2)
        self.cnn2 = CNNUnit((4, 16), num_layers=2, kernel_size=3, stride=2, activation=activation)
        self.pooling2 = nn.MaxPool2d(2)
        self.mlp = MultiLayerPerceptron((self.input_size, 256, 256), 3, activation=activation)

    def forward(self, x):
        x = self.cnn1(x)        # (N, 4, 32, 32)
        x = self.pooling1(x)    # (N, 4, 16, 16)
        x = self.cnn2(x)        # (N, 16, 8, 8)
        x = self.pooling2(x)    # (N, 16, 4, 4)
        # x = self.mlp(x)
        # x = x.view(x.shape[0], -1)  # (N, 256)
        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),  # (N, 32, 32, 32)
            nn.MaxPool2d(2),    # (N, 32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(2),    # (N, 64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(2),     # (N, 64, 4, 4)
            nn.Flatten(),
            nn.Linear(1024, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 1024),
            Reshape(-1, 64, 4, 4),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)     # (N, 64)
        x = self.decoder(x)     # (N, 1, 64, 64)
        return x


class Autoencoder2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            MultiLayerPerceptron((32*32, 256, 256), 3, nn.ReLU()),
            nn.ReLU(),
            MultiLayerPerceptron((256, 256, 128), 3, nn.ReLU())
        )

        self.decoder = torch.nn.Sequential(
            MultiLayerPerceptron((128, 256, 256), 3, nn.ReLU()),
            nn.ReLU(),
            MultiLayerPerceptron((256, 256, 32*32), 3, nn.ReLU()),
            Reshape(-1, 1, 32, 32),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
