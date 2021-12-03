from Optimizer.chomp import chomp_grad
import torch


def chompy_partial_loss(q, par):
    (length_cost, collision_cost), (length_jac, collision_jac) = chomp_grad(q=q, par=par, jac_only=False,
                                                                            return_separate=True)
    length_jac = torch.from_numpy(length_jac)
    collision_jac = torch.from_numpy(collision_jac)
    return length_cost, collision_cost, length_jac, collision_jac


