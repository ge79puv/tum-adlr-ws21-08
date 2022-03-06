import numpy as np
import matplotlib.pyplot as plt

plot_path = './plot/arm_image/compare/'

r1 = np.load("arm_loss_image_global.npz")
train_feasible_history1 = r1["train_feasible_history"]
test_feasible_history1 = r1["test_feasible_history"]
train_loss_history1 = r1["train_loss_history"]
test_loss_history1 = r1["test_loss_history"]

r2 = np.load("arm_loss_flatten_global.npz")
train_feasible_history2 = r2["train_feasible_history"]
test_feasible_history2 = r2["test_feasible_history"]
train_loss_history2 = r2["train_loss_history"]
test_loss_history2 = r2["test_loss_history"]

r3 = np.load("arm_loss_image_relative.npz")
train_feasible_history3 = r3["train_feasible_history"]
test_feasible_history3 = r3["test_feasible_history"]
train_loss_history3 = r3["train_loss_history"]
test_loss_history3 = r3["test_loss_history"]

r4 = np.load("arm_loss_flatten_relative.npz")
train_feasible_history4 = r4["train_feasible_history"]
test_feasible_history4 = r4["test_feasible_history"]
train_loss_history4 = r4["train_loss_history"]
test_loss_history4 = r4["test_loss_history"]

plt.figure(1)
plt.plot(test_feasible_history1, label='global_image', color="green")
plt.plot(test_feasible_history2, label='global_flatten', color="red")
plt.plot(test_feasible_history3, label='relative_image', linestyle='--', color="green")
plt.plot(test_feasible_history4, label='relative_flatten', linestyle='--', color="red")

plt.title('test feasible rate')
plt.legend(loc="lower right")
plt.axis([0, None, 0.4, 0.6])
plt.savefig(plot_path + 'feasible')
plt.show()

plt.figure(2)
plt.plot(test_loss_history1, label='global_image', color="green")
plt.plot(test_loss_history2, label='global_flatten', color="red")
plt.plot(test_loss_history3, label='relative_image', linestyle='--', color="green")
plt.plot(test_loss_history4, label='relative_flatten', linestyle='--', color="red")

plt.title('test loss')
plt.legend()
plt.savefig(plot_path + 'loss')
plt.show()

