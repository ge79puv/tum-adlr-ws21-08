import torch


def scaling(x, input_limits, output_limits):
    input_center = (input_limits[:, 1] - input_limits[:, 0]) / 2 + input_limits[:, 0]
    output_center = (output_limits[:, 1] - output_limits[:, 0]) / 2 + output_limits[:, 0]

    factor = (output_limits[:, 1] - output_center) / (input_limits[:, 1] - input_center)
    x_scaled = (x - input_center) * factor + output_center
    return x_scaled


class Processing:
    def __init__(self, world_limits):
        self.world_limits = torch.from_numpy(world_limits)
        self.normalization = torch.ones(self.world_limits.shape)
        self.normalization[:, 0] *= -1

    def preprocessing(self, q):
        q_normalized = scaling(q, self.world_limits, self.normalization)
        return q_normalized

    def postprocessing(self, q):
        q_world = scaling(q, self.normalization, self.world_limits)
        return q_world
