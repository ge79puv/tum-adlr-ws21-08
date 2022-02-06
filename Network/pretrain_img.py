import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from Network.worlds import Worlds
from Robots import SingleSphere02
from network import Autoencoder, Autoencoder2
from parameter import Parameter


radius = 0.3  # Size of the robot [m]
robot = SingleSphere02(radius=radius)
par = Parameter(robot=robot, obstacle_img='rectangle')

world_limits = np.array([[0, 10],  # x [m]
                         [0, 10]])  # y [m]
par.robot.limits = world_limits
n_obstacles = 5
min_max_obstacle_size_voxel = [3, 15]
n_voxels = (64, 64)

n_worlds_train = 1000
worlds_batch_train = 10
worlds_train = Worlds(n_worlds_train, n_obstacles, min_max_obstacle_size_voxel, n_voxels, par)
worlds_loader_train = DataLoader(worlds_train.dist_images, batch_size=worlds_batch_train, shuffle=True)

n_worlds_test = 10
worlds_batch_test = 10
worlds_test = Worlds(n_worlds_test, n_obstacles, min_max_obstacle_size_voxel, n_voxels, par)
worlds_loader_test = DataLoader(worlds_test.dist_images, batch_size=worlds_batch_test, shuffle=False)

# ============================== Image Pretrain =============================
loss_func = torch.nn.MSELoss()
ae = Autoencoder2()
op = torch.optim.SGD(ae.parameters(), lr=0.001, momentum=0.9)

min_loss = np.inf
stop = 0
train_loss_history = []
for epoch in range(2000):
    train_loss = 0
    for i, (dist_imgs, idx) in enumerate(worlds_loader_train):
        predict = ae(dist_imgs)
        loss = loss_func(predict, dist_imgs) * 1000

        op.zero_grad()
        loss.backward()
        op.step()

        train_loss += loss
        if epoch % 20 == 0:
            if i == 0:
                plt.figure()
                plt.imshow(dist_imgs[0][0].detach().numpy(), origin='lower')
                plt.show()
                plt.figure()
                plt.imshow(predict[0][0].detach().numpy(), origin='lower')
                plt.show()

    train_loss_history.append(train_loss / n_worlds_train)
    print(epoch, "  ", train_loss / n_worlds_train)

    if train_loss < min_loss:
        min_loss = train_loss
        stop = 0
    elif epoch > 500:
        stop = stop + 1
        if stop > 10:
            break

torch.save(ae.encoder, "encoder")

plt.show()

plt.figure()
plt.plot(train_loss_history[3:], label='training')
# plt.plot(test_loss_history, label='test')
plt.title('loss')
plt.legend()
plt.savefig('loss')
plt.show()
