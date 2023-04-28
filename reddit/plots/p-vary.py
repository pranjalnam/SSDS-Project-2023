import matplotlib.pyplot as plt
import numpy as np

plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('axes', labelsize='xx-large', titlesize='xx-large')
plt.rcParams['legend.title_fontsize'] = 'xx-large'
plt.rcParams['legend.fontsize'] = 'medium'

data_reddit = [
    [[45.01, 95.06, 144.73, 194.54, 244.12, 293.97, 343.44, 393.23, 442.91, 492.4, 541.93, 591.48],
     [0.4368, 0.5206, 0.6813, 0.7451, 0.8111, 0.8417, 0.8672, 0.892, 0.906, 0.9121, 0.9162, 0.9189],
     [2.9071, 2.329, 1.8934, 1.535, 1.2645, 1.0693, 0.9329, 0.8355, 0.7631, 0.7073, 0.6633, 0.6275]],

    [[46.28, 96.0, 145.6, 196.31, 247.32, 298.45, 348.59, 399.06, 449.02],
     [0.4988, 0.6718, 0.7493, 0.8131, 0.8513, 0.8736, 0.894, 0.9082, 0.9165],
     [2.741, 2.0575, 1.5818, 1.258, 1.0479, 0.9044, 0.809, 0.7414, 0.6924]],

    [[45.02, 96.73, 146.52, 197.18, 246.89, 297.05, 347.72, 398.57, 448.45, 499.22],
     [0.4694, 0.6438, 0.731, 0.7991, 0.8382, 0.8638, 0.8828, 0.8944, 0.9057, 0.9125],
     [2.7102, 2.0779, 1.6528, 1.3514, 1.1389, 0.9882, 0.8788, 0.8004, 0.7427, 0.6982]],

    [[45.25, 99.14, 151.12, 201.27, 252.75, 302.94, 352.58, 403.27, 454.98, 506.53, 556.72, 606.82, 657.79],
     [0.3888, 0.5172, 0.6953, 0.7607, 0.8186, 0.8446, 0.8638, 0.8761, 0.8912, 0.9019, 0.9074, 0.9115, 0.9152],
     [2.8786, 2.3315, 1.8769, 1.515, 1.2706, 1.0921, 0.9734, 0.8801, 0.8099, 0.7566, 0.7144, 0.6812, 0.6519]],

[[50.06, 101.61, 150.98, 201.8, 251.16, 300.69, 350.83, 400.76, 451.43, 501.39, 551.55], [0.3831, 0.5667, 0.6192, 0.693, 0.7823, 0.8118, 0.8532, 0.8821, 0.899, 0.9083, 0.9138], [2.9907, 2.3104, 1.8984, 1.5881, 1.3323, 1.1359, 0.9993, 0.8967, 0.8195, 0.7622, 0.7155]]

]

# create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))


# plot the data on the subplots and add grid lines
axs[0].plot(data_reddit[0][0], data_reddit[0][2], 'orange', linewidth=1.5, label="P = 1")
axs[0].plot(data_reddit[1][0], data_reddit[1][2], 'crimson', linewidth=1.5, label="P = 2")
axs[0].plot(data_reddit[2][0], data_reddit[2][2], 'forestgreen', linewidth=1.5, label="P = 3")
axs[0].plot(data_reddit[3][0], data_reddit[3][2], 'navy', linewidth=1.5, label="P = 4")
axs[0].plot(data_reddit[4][0], data_reddit[4][2], 'magenta', linewidth=1.5, label="P = 5")

axs[0].set_xlabel('Wall Time (s)')
axs[0].set_ylabel('Training Loss')
# axs[0].set_ylim([0, 40])
axs[0].axhline(y = 0.7, color = 'k', linestyle = '--', linewidth=1.5)
axs[0].grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs[0].grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs[0].minorticks_on()


axs[1].plot(data_reddit[0][0], data_reddit[0][1], 'orange', linewidth=1.5, label="P = 1")
axs[1].plot(data_reddit[1][0], data_reddit[1][1], 'crimson', linewidth=1.5, label="P = 2")
axs[1].plot(data_reddit[2][0], data_reddit[2][1], 'forestgreen', linewidth=1.5, label="P = 3")
axs[1].plot(data_reddit[3][0], data_reddit[3][1], 'navy', linewidth=1.5, label="P = 4")
axs[1].plot(data_reddit[4][0], data_reddit[4][1], 'magenta', linewidth=1.5, label="P = 5")


axs[1].set_xlabel('Wall Time (s)')
axs[1].set_ylabel('Test Accuracy')
# axs[1].set_xlim([0, 400])
axs[1].axhline(y = 0.89, color = 'k', linestyle = '--', linewidth=1.5)
axs[1].grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs[1].grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs[1].minorticks_on()

# add a title for the entire figure
fig.suptitle('Convergence Plots v P (mw-PR on Reddit)', fontsize='xx-large')
axs[0].set_title('Loss Plot')
axs[1].set_title('Acc Plot')
axs[0].legend()
axs[1].legend()
# adjust spacing between subplots
fig.subplots_adjust(hspace=0.6, wspace=0.4)
plt.tight_layout()
# display the plot
plt.savefig("reddit-p-vary.jpg", dpi=600)
plt.show()
