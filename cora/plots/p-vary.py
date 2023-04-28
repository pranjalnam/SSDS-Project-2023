import matplotlib.pyplot as plt
import numpy as np

plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('axes', labelsize='xx-large', titlesize='xx-large')
plt.rcParams['legend.title_fontsize'] = 'xx-large'
plt.rcParams['legend.fontsize'] = 'medium'

data_cora = [
]

# create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# plot the data on the subplots and add grid lines
axs[0].plot(data_cora[0][0], data_cora[0][2], 'orange', linewidth=1.5, label="P = 1")
axs[0].plot(data_cora[1][0], data_cora[1][2], 'crimson', linewidth=1.5, label="P = 2")
axs[0].plot(data_cora[2][0], data_cora[2][2], 'forestgreen', linewidth=1.5, label="P = 3")
axs[0].plot(data_cora[3][0], data_cora[3][2], 'navy', linewidth=1.5, label="P = 4")
axs[0].plot(data_cora[4][0], data_cora[4][2], 'magenta', linewidth=1.5, label="P = 5")

axs[0].set_xlabel('Wall Time (s)')
axs[0].set_ylabel('Training Loss')
# axs[0].set_ylim([0, 40])
axs[0].axhline(y=0.6, color='k', linestyle='--', linewidth=1.5)
axs[0].grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs[0].grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs[0].minorticks_on()

axs[1].plot(data_cora[0][0], data_cora[0][1], 'orange', linewidth=1.5, label="P = 1")
axs[1].plot(data_cora[1][0], data_cora[1][1], 'crimson', linewidth=1.5, label="P = 2")
axs[1].plot(data_cora[2][0], data_cora[2][1], 'forestgreen', linewidth=1.5, label="P = 3")
axs[1].plot(data_cora[3][0], data_cora[3][1], 'navy', linewidth=1.5, label="P = 4")
axs[1].plot(data_cora[4][0], data_cora[4][1], 'magenta', linewidth=1.5, label="P = 5")

axs[1].set_xlabel('Wall Time (s)')
axs[1].set_ylabel('Test Accuracy')
# axs[1].set_ylim([0.85, 1])
axs[1].axhline(y=0.91, color='k', linestyle='--', linewidth=1.5)
axs[1].grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs[1].grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs[1].minorticks_on()

# add a title for the entire figure
fig.suptitle('Convergence Plots v P (mw-PR on MNIST)', fontsize='xx-large')
axs[0].set_title('Loss Plot')
axs[1].set_title('Acc Plot')
axs[0].legend()
axs[1].legend()
# adjust spacing between subplots
fig.subplots_adjust(hspace=0.6, wspace=0.4)
plt.tight_layout()
# display the plot
plt.savefig("p-vary.jpg", dpi=600)
plt.show()
