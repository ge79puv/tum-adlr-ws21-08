import numpy as np
import torch
from torch.utils.data import Dataset


class StartEndPointsDataset(Dataset):
    def __init__(self, number, dof, image, world_limits, proc):
        self.number = number
        self.start_points = sample_points(number, dof, image, world_limits)  # (number, dof)
        self.end_points = sample_points(number, dof, image, world_limits)   # (number, dof)
        self.pairs = torch.cat((proc.preprocessing(self.start_points),
                                proc.preprocessing(self.end_points)),
                               1)  # (number, 2 * dof)

    def __len__(self):
        return self.number

    def __getitem__(self, item):
        return self.start_points[item], self.end_points[item], self.pairs[item]


def sample_points(number, dof, image, world_limits):
    def sample(invalid):
        q_attempt = np.random.rand(invalid, dof)
        q_attempt_voxel = (q_attempt * image.shape).astype(int)
        mask = image[q_attempt_voxel[:, 0], q_attempt_voxel[:, 1]]
        if mask.sum() > 0:
            new = sample(mask.sum())
            q_attempt[mask] = new
        else:
            return q_attempt
        return q_attempt

    q_0 = sample(number)
    q_sampled = world_limits[:, 0] + q_0 * world_limits[:, 1]
    return torch.FloatTensor(q_sampled)
