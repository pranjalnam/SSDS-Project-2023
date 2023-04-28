import matplotlib.pyplot as plt
import numpy as np

plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('axes', labelsize='xx-large', titlesize='xx-large')
plt.rcParams['legend.title_fontsize'] = 'xx-large'
plt.rcParams['legend.fontsize'] = 'medium'


# create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))


data = [
    [[45.32, 94.41, 143.04, 191.9, 240.64, 293.24, 342.31, 394.69, 443.9, 492.81, 541.76, 590.83, 640.33, 692.61, 741.21, 793.75], [0.5443, 0.7062, 0.7816, 0.8422, 0.8827, 0.9005, 0.9089, 0.9152, 0.9181, 0.9199, 0.9214, 0.9228, 0.9238, 0.925, 0.9261, 0.9269], [2.5026, 1.7609, 1.3428, 1.0991, 0.9381, 0.8346, 0.7675, 0.7144, 0.677, 0.6474, 0.6218, 0.6002, 0.5795, 0.5599, 0.5436, 0.5276]],
    [[45.21, 95.32, 144.75, 194.17, 247.18, 295.96, 345.11, 394.17, 443.45, 493.26, 542.17, 592.54, 641.68, 694.35, 747.45, 799.94, 852.49, 904.94, 957.57, 1009.99, 1063.93, 1116.78], [0.3189, 0.5391, 0.6305, 0.6724, 0.7349, 0.7578, 0.7934, 0.8222, 0.8389, 0.8513, 0.8614, 0.8712, 0.8835, 0.8955, 0.9036, 0.9079, 0.9118, 0.9152, 0.9178, 0.9195, 0.9212, 0.9228], [3.0269, 2.5375, 2.1862, 1.8951, 1.6478, 1.4582, 1.3155, 1.1977, 1.1005, 1.0202, 0.9535, 0.8946, 0.8423, 0.7944, 0.7544, 0.7207, 0.6921, 0.6676, 0.6462, 0.6275, 0.6103, 0.5945]],

    [[46.06, 95.21, 144.05, 193.48, 242.59, 292.05, 341.48, 391.19, 440.85, 489.68, 538.89, 588.89, 637.9, 686.74, 736.62], [0.4016, 0.6562, 0.7217, 0.7742, 0.8235, 0.856, 0.8829, 0.9044, 0.913, 0.9176, 0.9212, 0.9239, 0.9257, 0.9274, 0.9284], [2.9169, 2.2526, 1.7353, 1.3714, 1.1281, 0.9636, 0.8504, 0.7696, 0.709, 0.6631, 0.6262, 0.5959, 0.5713, 0.5505, 0.5323]],

    [[46.23, 95.45, 144.57, 193.86, 243.01, 292.03, 341.08, 390.22, 439.0, 492.51, 541.36, 591.38, 640.5, 689.48, 738.57, 787.37], [0.4953, 0.5897, 0.6557, 0.6965, 0.7502, 0.8037, 0.8238, 0.8465, 0.8624, 0.8787, 0.8853, 0.8925, 0.9019, 0.9107, 0.9165, 0.9197], [2.7586, 2.2652, 1.9875, 1.835, 1.5589, 1.332, 1.1712, 1.0383, 0.9397, 0.8242, 0.7855, 0.7532, 0.7139, 0.6811, 0.6553, 0.6323]],
    # [[45.54, 95.78, 145.21, 194.03, 243.46, 291.87, 342.58, 390.98, 439.92, 489.17, 537.9, 587.46, 636.12, 685.27, 734.87], [0.4233, 0.6159, 0.696, 0.7605, 0.814, 0.8482, 0.8824, 0.9007, 0.9107, 0.916, 0.919, 0.9212, 0.9235, 0.9252, 0.9264], [2.8073, 2.1497, 1.6964, 1.3771, 1.1513, 0.9959, 0.8807, 0.8018, 0.74, 0.6931, 0.6564, 0.6272, 0.6028, 0.5823, 0.5654]]

[[45.64, 95.43, 145.9, 195.75, 245.17, 294.8, 345.22, 395.93, 446.19, 496.11], [0.4738, 0.6529, 0.7315, 0.8024, 0.8382, 0.8871, 0.906, 0.9133, 0.9175, 0.9202], [2.8072, 2.1029, 1.6144, 1.2661, 1.0395, 0.8997, 0.8076, 0.7425, 0.6951, 0.6572]]
]

# plot the data on the subplots and add grid lines
axs[0].plot(data[0][0], data[0][2], 'mediumvioletred', linewidth=1.5, label="AR")
axs[0].plot(data[1][0], data[1][2], 'red', linewidth=1.5, label="s-AR")
# axs[0].plot(data[2][0], data[2][2], 'darkgoldenrod', linewidth=1.5, label="g-arar")
# axs[0].plot(data[3][0], data[3][2], 'navy', linewidth=1.5, label="g-arps")
# axs[0].plot(data[4][0], data[4][2], 'forestgreen', linewidth=1.5, label="p-reduce (P=2)")

axs[0].set_xlabel('Wall Time (s)')
axs[0].set_ylabel('Training Loss')
# axs[0].set_ylim([0, 4])
axs[0].axhline(y = 0.6, color = 'k', linestyle = '--', linewidth=1.5)
axs[0].grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs[0].grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs[0].minorticks_on()


axs[1].plot(data[0][0], data[0][1], 'mediumvioletred', linewidth=1.5, label="AR")
axs[1].plot(data[1][0], data[1][1], 'red', linewidth=1.5, label="s-AR")
# axs[1].plot(data[2][0], data[2][1], 'darkgoldenrod', linewidth=1.5, label="g-arar")
# axs[1].plot(data[3][0], data[3][1], 'navy', linewidth=1.5, label="g-arps")
# axs[1].plot(data[4][0], data[4][1], 'forestgreen', linewidth=1.5, label="p-reduce (P=2)")

axs[1].set_xlabel('Wall Time (s)')
axs[1].set_ylabel('Test Accuracy')
# axs[1].set_ylim([0.85, 1])
axs[1].axhline(y = 0.91 , color = 'k', linestyle = '--', linewidth=1.5)
axs[1].axvline(x=np.interp(0.91, data[0][1], data[0][0]),linestyle ='--', ymin=0.0, ymax=0.92, alpha=0.3, color='mediumvioletred')
axs[1].axvline(x=np.interp(0.91, data[1][1], data[1][0]),linestyle ='--', ymin=0.0, ymax=0.92, alpha=0.3, color='red')
# axs[1].axvline(x=np.interp(0.91, data[2][1], data[2][0]),linestyle ='--', ymin=0.0, ymax=0.92, alpha=0.3, color='darkgoldenrod')
# axs[1].axvline(x=np.interp(0.91, data[3][1], data[3][0]),linestyle ='--', ymin=0.0, ymax=0.92, alpha=0.3, color='navy')
# axs[1].axvline(x=np.interp(0.91, data[4][1], data[4][0]),linestyle ='--', ymin=0.0, ymax=0.92, alpha=0.3, color='forestgreen')

axs[1].grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs[1].grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs[1].minorticks_on()

# add a title for the entire figure
fig.suptitle('Straggler Convergence (Reddit)', fontsize='xx-large')
axs[0].set_title('Loss Plot')
axs[1].set_title('Acc Plot')
axs[0].legend()
axs[1].legend()
# adjust spacing between subplots
fig.subplots_adjust(hspace=0.6, wspace=0.4)
plt.tight_layout()
# display the plot
plt.savefig("reddit-straggler-effect.jpg", dpi=600)
plt.show()
