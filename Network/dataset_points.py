import numpy as np
import torch
from torch.utils.data import Dataset

from Optimizer import feasibility_check


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

    def get_collision_rate(self):
        waypoints = np.linspace(self.start_points, self.end_points, 10).swapaxes(0, 1)
        status = feasibility_check(waypoints[:, 1:-1, :], self.par)
        print("collision rate: ", (status == -1).mean())

    def set_collision_rate(self, collision_rate):
        waypoints = np.linspace(self.start_points, self.end_points, 10).swapaxes(0, 1)
        status = feasibility_check(waypoints[:, 1:-1, :], self.par)
        diff_number = int(self.start_points[status == -1].shape[0] - self.number * collision_rate)

        if diff_number > 0:     # want more non collision pairs
            to_update_start_points = self.start_points[status == -1]
            to_update_end_points = self.end_points[status == -1]
            new_start_points, new_end_points = filter_collision(
                to_update_start_points[:diff_number],
                to_update_end_points[:diff_number], self.par)

            to_update_start_points[:diff_number] = new_start_points
            to_update_end_points[:diff_number] = new_end_points
            self.start_points[status == -1] = to_update_start_points
            self.end_points[status == -1] = to_update_end_points

        elif diff_number < 0:   # want more collision pairs
            to_update_start_points = self.start_points[status == 1]
            to_update_end_points = self.end_points[status == 1]
            new_start_points, new_end_points = filter_non_collision(
                to_update_start_points[:-diff_number],
                to_update_end_points[:-diff_number], self.par)

            to_update_start_points[:-diff_number] = new_start_points
            to_update_end_points[:-diff_number] = new_end_points
            self.start_points[status == 1] = to_update_start_points
            self.end_points[status == 1] = to_update_end_points

        # waypoints = np.linspace(self.start_points, self.end_points, 10).swapaxes(0, 1)
        # status = feasibility_check(waypoints[:, 1:-1, :], self.par)
        # print("collision rate: ", (status == -1).mean())

    def set_min_distance(self, min_distance):
        distance = ((self.start_points - self.end_points) ** 2).sum(axis=1).detach().numpy()
        invalid = distance < min_distance ** 2
        self.start_points[invalid], self.end_points[invalid] = filter_short(self.start_points[invalid],
                                                                            self.end_points[invalid],
                                                                            min_distance, self.par)

        # distance = ((self.start_points - self.end_points) ** 2).sum(axis=1).detach().numpy()
        # invalid = distance < min_distance ** 2
        # print("distance shorter than min: ", invalid.sum())


def filter_short(start_points, end_points, min_distance, par):
    distance = ((start_points - end_points) ** 2).sum(axis=1).detach().numpy()
    invalid = distance < min_distance ** 2
    print(start_points[invalid].shape[0])
    if start_points[invalid].shape[0] > 0:
        new = StartEndPointsDataset(start_points[invalid].shape[0], par)
        start_points[invalid], end_points[invalid] = filter_short(new.start_points, new.end_points, min_distance, par)
    else:
        return start_points, end_points
    return start_points, end_points


def filter_collision(start_points, end_points, par):
    waypoints = np.linspace(start_points, end_points, 10).swapaxes(0, 1)
    status = feasibility_check(waypoints[:, 1:-1, :], par)
    if start_points[status == 1].shape[0] < start_points.shape[0]:  # not all non collision
        new = StartEndPointsDataset(start_points[status == -1].shape[0], par)
        start_points[status == -1], end_points[status == -1] = filter_collision(new.start_points, new.end_points, par)
    else:
        return start_points, end_points
    return start_points, end_points


def filter_non_collision(start_points, end_points, par):
    waypoints = np.linspace(start_points, end_points, 10).swapaxes(0, 1)
    status = feasibility_check(waypoints[:, 1:-1, :], par)
    if start_points[status == -1].shape[0] < start_points.shape[0]:  # not all collision
        new = StartEndPointsDataset(start_points[status == 1].shape[0], par)
        start_points[status == 1], end_points[status == 1] = filter_non_collision(new.start_points, new.end_points, par)
    else:
        return start_points, end_points
    return start_points, end_points


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
