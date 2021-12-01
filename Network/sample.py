import numpy as np
import torch


def sample_points(number, dof, image, limits):
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
    q_sampled = limits[:, 0] + q_0 * limits[:, 1]
    return torch.FloatTensor(q_sampled)


def normalization(x):

    pass
