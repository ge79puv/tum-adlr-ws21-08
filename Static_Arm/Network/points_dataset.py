import numpy as np
import torch
from torch.utils.data import Dataset

from chompy.Optimizer import feasibility_check


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


def sample_points(number, par):
    def sample(invalid):
        q_attempt = np.random.rand(invalid, par.robot.n_dof) * (par.robot.limits[:, 1] - par.robot.limits[:, 0])
        # (5990, 2)
        status = feasibility_check(q_attempt[:, np.newaxis, :], par)  # (5990, 1, 2)
        # (5990,)  [-1,1]
        if q_attempt[status == -1].shape[0] > 0:
            new = sample(q_attempt[status == -1].shape[0])
            q_attempt[status == -1] = new
        else:
            return q_attempt
        return q_attempt

    q_sampled = sample(number)  # (5990, 2)
    return torch.FloatTensor(q_sampled)


class BasePointsDataset(Dataset):
    def __init__(self, number, par):
        self.number = number
        self.start_points = sample_points_arm(number, par)[:, 0, :]  # (number, dof)
        self.end_points = sample_points_arm(number, par)[:, 1, :]   # (number, dof)
        self.par = par

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

        q = par.robot.sample_q((invalid, 2))  # will not repeat, different number different value  (number,2,dof)
        status = feasibility_check(q, par)  # (number, ) [-1,1]
        # print(q[status == -1].shape[0])
        if q[status == -1].shape[0] > 0:
            new = sample(q[status == -1].shape[0])
            q[status == -1] = new
        else:
            return q
        return q

    q_sampled = sample(number)  # (number,2,dof)
    return torch.FloatTensor(q_sampled)


'''
x = par.robot.get_x_spheres(q)  # (number, 2, number_sphere, 2)
# print(x.shape)

# Todo 验证x的位置有没有和obstacles重合的
x = np.reshape(x, (-1, 2))  # (number * 2 * number_sphere, 2)
# print(x[:, np.newaxis, :].shape)
status = feasibility_check(x[:, np.newaxis, :], par)  # (number * 2 * number_sphere, ) [-1,1]
q[np.any(status == -1, axis=1)]
'''
