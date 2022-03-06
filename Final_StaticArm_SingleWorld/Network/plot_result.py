import numpy as np
import matplotlib.pyplot as plt

plot_path = './plot/compare/'

r1 = np.load("arm_loss_global.npz")
train_feasible_history1 = r1["train_feasible_history"]
test_feasible_history1 = r1["test_feasible_history"]
train_loss_history1 = r1["train_loss_history"]
test_loss_history1 = r1["test_loss_history"]

r2 = np.load("arm_loss_relative.npz")
train_feasible_history2 = r2["train_feasible_history"]
test_feasible_history2 = r2["test_feasible_history"]
train_loss_history2 = r2["train_loss_history"]
test_loss_history2 = r2["test_loss_history"]

plt.figure(1)
plt.plot(train_feasible_history1, label='train_global', color="blue")
plt.plot(train_feasible_history2, label='train_relative', color="orange")
plt.plot(test_feasible_history1, label='test_global', linestyle='--', color="blue")
plt.plot(test_feasible_history2, label='test_relative', linestyle='--', color="orange")

plt.title('feasible rate')
plt.legend(loc="lower right")
plt.axis([0, None, 0.4, 0.8])
plt.savefig(plot_path + 'feasible')
plt.show()

plt.figure(2)
plt.plot(train_loss_history1, label='train_global', color="blue")
plt.plot(train_loss_history2, label='train_relative', color="orange")
plt.plot(test_loss_history1, label='test_global', linestyle='--', color="blue")
plt.plot(test_loss_history2, label='test_relative', linestyle='--', color="orange")

plt.title('loss')
plt.legend()
plt.axis([0, None, 0, 4])
plt.savefig(plot_path + 'loss')
plt.show()
