import numpy as np
import torch
from torch.utils.data import Dataset

from chompy.Optimizer import feasibility_check
from helper import joints_preprocessing


class BasePointsDataset(Dataset):
    def __init__(self, number, par):
        self.number = number
        self.start_points = joints_preprocessing(sample_points_arm(number, par)[:number, :])  # (number, dof)
        self.end_points = joints_preprocessing(sample_points_arm(number, par)[number:, :])  # (number, dof)

    def __len__(self):
        return self.number

    def __getitem__(self, item):
        return self.start_points[item], self.end_points[item]

    def add_data(self, SEDataset):
        self.number = self.number + SEDataset.number
        self.start_points = torch.cat((self.start_points, SEDataset.start_points), 0)
        self.end_points = torch.cat((self.end_points, SEDataset.end_points), 0)


def sample_points_arm(number, par):

    def sample(invalid):

        q = par.robot.sample_q(invalid)  # will not repeat, different number different value  (number,dof)
        # print(q.shape)
        status = feasibility_check(q[:, np.newaxis, :], par)  # (number, ) [-1,1]   (number,1,dof)
        # print(q[status == -1].shape[0])

        if q[status == -1].shape[0] > 0:
            new = sample(q[status == -1].shape[0])
            q[status == -1] = new
        else:
            return q
        return q

    q_sampled = sample(2*number)  # (2*number,dof)
    return torch.FloatTensor(q_sampled)


