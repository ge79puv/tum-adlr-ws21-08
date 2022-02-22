import numpy as np
from matplotlib import pyplot as plt


global_loss = np.load('plot/path_repr/world0/global/loss_history.npz')['test_loss_history']
global_length = np.load('plot/path_repr/world0/global/loss_history.npz')['test_length_loss_history']
global_feasible = np.load('plot/path_repr/world0/global/loss_history.npz')['test_feasible_history']

nurbs_loss = np.load('plot/path_repr/world0/nurbs/loss_history.npz')['test_loss_history']
nurbs_length = np.load('plot/path_repr/world0/nurbs/loss_history.npz')['test_length_loss_history']
nurbs_feasible = np.load('plot/path_repr/world0/nurbs/loss_history.npz')['test_feasible_history']

relative_loss = np.load('plot/path_repr/world0/relative/loss_history.npz')['test_loss_history']
relative_length = np.load('plot/path_repr/world0/relative/loss_history.npz')['test_length_loss_history']
relative_feasible = np.load('plot/path_repr/world0/relative/loss_history.npz')['test_feasible_history']

print("global loss: ", global_loss[20:].mean())
print("global length loss: ", global_length[20:].mean())
print("global feasibility: ", global_feasible[20:].mean())

print("nurbs loss: ", nurbs_loss[20:].mean())
print("nurbs length loss: ", nurbs_length[20:].mean())
print("nurbs feasibility: ", nurbs_feasible[20:].mean())

print("relative loss: ", relative_loss[20:].mean())
print("relative length loss: ", relative_length[20:].mean())
print("relative feasibility: ", relative_feasible[20:].mean())

plt.plot(global_feasible, label='global')
plt.plot(nurbs_feasible, label='nurbs')
plt.plot(relative_feasible, label='relative')
plt.title('feasible rate')
plt.legend()
plt.axis([0, None, 0, 1])
plt.savefig('path_repr_feasible')
plt.show()

plt.plot(global_loss, label='global')
plt.plot(nurbs_loss, label='nurbs')
plt.plot(relative_loss, label='relative')
plt.plot(global_length, '--', label='global_length')
plt.plot(nurbs_length, '--', label='nurbs_length')
plt.plot(relative_length, '--', label='relative_length')
plt.title('loss')
plt.axis([10, None, None, 5])
plt.legend()
plt.savefig('path_repr_loss')
plt.show()
