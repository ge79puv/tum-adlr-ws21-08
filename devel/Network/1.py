import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from chompy.GridWorld import create_rectangle_image
from chompy.Kinematic.Robots import SingleSphere02
from points_dataset import StartEndPointsDataset
from helper import Processing
from loss_function import chompy_partial_loss
from network import Backbone2D, Dummy
from visualization import plot_paths
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter, initialize_oc


np.random.seed(10)
device = torch.device("cpu")            # "cuda:0" if torch.cuda.is_available() else
print(device)
plot_path = './plot/o10/BothJacAfter30C5/'
os.makedirs(plot_path, exist_ok=True)

y1 = torch.load("./myTensor1.pt")
y2 = torch.load("./myTensor2.pt")
y3 = torch.load("./myTensor3.pt")


plt.figure()
plt.plot(y1, label='1:1')
plt.plot(y2, label='1:5')
plt.plot(y3, label='1:10')
plt.title('test feasibility rate')
plt.ylim(0, 1)
plt.xlim(0, 100)
plt.legend()
plt.savefig(plot_path + 'feasibility')
plt.show()




