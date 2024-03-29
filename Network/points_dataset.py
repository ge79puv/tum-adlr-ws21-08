import numpy as np
import torch
from torch.utils.data import Dataset

from Optimizer import feasibility_check


def sample_points(number, par):
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


class StartEndPointsDataset(Dataset):
    def __init__(self, number, par):
        self.number = number
        self.start_points = sample_points(number, par)  # (number, dof)
        self.end_points = sample_points(number, par)   # (number, dof)
        self.par = par

    def __len__(self):
        return self.number

    def __getitem__(self, item):
        return self.start_points[item], self.end_points[item]

    def add_data(self, SEDataset):
        self.number = self.number + SEDataset.number
        self.start_points = torch.cat((self.start_points, SEDataset.start_points), 0)
        self.end_points = torch.cat((self.end_points, SEDataset.end_points), 0)



