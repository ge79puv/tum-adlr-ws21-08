import numpy as np
import torch
from torch.utils.data import Dataset

from Optimizer import feasibility_check


class StartEndPointsDataset(Dataset):
    def __init__(self, number, image, par, proc):
        self.number = number
        self.start_points = sample_points(number, image, par)  # (number, dof)
        self.end_points = sample_points(number, image, par)   # (number, dof)
        self.pairs = torch.cat((proc.preprocessing(self.start_points),
                                proc.preprocessing(self.end_points)),
                               1)  # (number, 2 * dof)

    def __len__(self):
        return self.number

    def __getitem__(self, item):
        return self.start_points[item], self.end_points[item], self.pairs[item]

    def add_data(self, SEDataset):
        self.number = self.number + SEDataset.number
        self.start_points = torch.cat((self.start_points, SEDataset.start_points), 0)
        self.end_points = torch.cat((self.end_points, SEDataset.end_points), 0)
        self.pairs = torch.cat((self.pairs, SEDataset.pairs), 0)


def sample_points(number, image, par):
    def sample(invalid):
        q_attempt = np.random.rand(invalid, par.robot.n_dof) * (par.robot.limits[:, 1] - par.robot.limits[:, 0])
        status = feasibility_check(q_attempt[:, np.newaxis, :], par)
        if q_attempt[status == -1].shape[0] > 0:
            new = sample(q_attempt[status == -1].shape[0])
            q_attempt[status == -1] = new
        else:
            return q_attempt
        return q_attempt

    q_sampled = sample(number)
    return torch.FloatTensor(q_sampled)
