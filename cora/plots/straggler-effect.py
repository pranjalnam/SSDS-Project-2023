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
[[45.66, 95.26, 143.46, 191.11, 238.41, 286.35, 334.01, 382.11, 430.0, 478.27, 526.23, 574.44, 621.75, 669.84], [0.0468, 0.0637, 0.1915, 0.3003, 0.3894, 0.4611, 0.516, 0.5537, 0.582, 0.599, 0.6105, 0.6221, 0.6283, 0.6351], [4.1023, 3.829, 3.5166, 3.1793, 2.8512, 2.5475, 2.2955, 2.0968, 1.9446, 1.8324, 1.7472, 1.6821, 1.6332, 1.5939]],
[[46.61, 96.93, 145.38, 194.06, 242.41, 291.02, 339.6, 388.46, 437.2, 485.66, 534.27, 582.88, 631.98, 679.19, 727.84, 776.46, 824.82, 873.45, 922.3, 970.72, 1019.28, 1067.14, 1114.72], [0.0483, 0.0463, 0.0472, 0.0987, 0.1634, 0.2541, 0.305, 0.3567, 0.4138, 0.4549, 0.4902, 0.5175, 0.5396, 0.5596, 0.5763, 0.5893, 0.5987, 0.6076, 0.6125, 0.6216, 0.6271, 0.6292, 0.6335], [4.1866, 4.0572, 3.9008, 3.7394, 3.5598, 3.3642, 3.1607, 2.9582, 2.7648, 2.5858, 2.4248, 2.2834, 2.1609, 2.0603, 1.9702, 1.8941, 1.8297, 1.7753, 1.7292, 1.6902, 1.6568, 1.6295, 1.6059]],
# [[45.72, 94.06, 142.06, 190.35, 238.53, 286.58, 334.63, 382.89, 430.95, 478.94, 527.23, 575.2, 623.15, 671.69, 720.13, 768.14, 816.42, 864.29, 912.37, 960.59, 1008.5], [0.0426, 0.1079, 0.148, 0.2024, 0.3168, 0.4143, 0.4732, 0.5126, 0.5413, 0.5606, 0.5768, 0.5896, 0.5992, 0.6069, 0.6117, 0.617, 0.6214, 0.6246, 0.6261, 0.6297, 0.6305], [4.2167, 4.0347, 3.7596, 3.4505, 3.1333, 2.8329, 2.5716, 2.3532, 2.1811, 2.0459, 1.9393, 1.8554, 1.7892, 1.737, 1.6959, 1.6637, 1.6387, 1.6194, 1.6048, 1.5942, 1.5869]],
# [[46.37, 94.68, 141.82, 190.1, 237.36, 284.71, 335.02, 382.84, 430.49, 478.4, 526.2, 574.12, 621.93, 669.8, 717.68], [0.0409, 0.047, 0.1379, 0.1942, 0.2551, 0.302, 0.4358, 0.481, 0.5032, 0.5362, 0.5711, 0.5914, 0.6059, 0.6159, 0.625], [4.1031, 3.9314, 3.7292, 3.496, 3.3421, 3.196, 2.764, 2.5341, 2.3901, 2.2181, 2.0834, 1.9634, 1.8789, 1.8066, 1.7509]],
# [[45.74, 93.16, 141.24, 188.92, 235.89, 283.79, 331.99, 379.95, 427.03, 474.14, 521.85, 569.06, 606.03, 633.33, 673.99, 709.03], [0.0904, 0.0898, 0.0982, 0.238, 0.3553, 0.4338, 0.4917, 0.5349, 0.564, 0.5808, 0.5911, 0.6309, 0.6229, 0.6229, 0.6309, 0.6309], [4.1292, 3.8723, 3.6087, 3.3489, 3.0675, 2.7817, 2.5176, 2.2894, 2.1162, 1.9865, 1.889, 1.8112, 1.7525, 1.7017, 1.6607, 1.6273]]
]


# plot the data on the subplots and add grid lines
axs[0].plot(data[0][0], data[0][2], 'mediumvioletred', linewidth=1.5, label="AR")
axs[0].plot(data[1][0], data[1][2], 'red', linewidth=1.5, label="s-AR")
# axs[0].plot(data[2][0], data[2][2], 'darkgoldenrod', linewidth=1.5, label="g-ARAR")
# axs[0].plot(data[3][0], data[3][2], 'navy', linewidth=1.5, label="g-ARPS")
# axs[0].plot(data[4][0], data[4][2], 'forestgreen', linewidth=1.5, label="mw-PR (P=2)")
# axs[0].plot(data[5][0], data[5][2], 'orange', linewidth=1.5, label="p-reduce (P=2)")
axs[0].set_xlabel('Wall Time (s)')
axs[0].set_ylabel('Training Loss')
# axs[0].set_ylim([0, 4])
axs[0].axhline(y = 1.7, color = 'k', linestyle = '--', linewidth=1.5)
axs[0].grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs[0].grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs[0].minorticks_on()


axs[1].plot(data[0][0], data[0][1], 'mediumvioletred', linewidth=1.5, label="AR")
axs[1].plot(data[1][0], data[1][1], 'red', linewidth=1.5, label="s-AR")
# axs[1].plot(data[2][0], data[2][1], 'darkgoldenrod', linewidth=1.5, label="g-ARAR")
# axs[1].plot(data[3][0], data[3][1], 'navy', linewidth=1.5, label="g-ARPS")
# axs[1].plot(data[4][0], data[4][1], 'forestgreen', linewidth=1.5, label="mw-PR (P=2)")
# axs[1].plot(data[5][0], data[5][1], 'orange', linewidth=1.5, label="p-reduce (P=2)")
axs[1].set_xlabel('Wall Time (s)')
axs[1].set_ylabel('Test Accuracy')
# axs[1].set_ylim([0.3, 0.6])
axs[1].axhline(y = 0.63 , color = 'k', linestyle = '--', linewidth=1.5)
axs[1].axvline(x=np.interp(0.65, data[0][1], data[0][0]),linestyle ='--', ymin=0.0, ymax=0.95, alpha=0.3, color='mediumvioletred')
axs[1].axvline(x=np.interp(0.65, data[1][1], data[1][0]),linestyle ='--', ymin=0.0, ymax=0.95, alpha=0.3, color='red')
# axs[1].axvline(x=np.interp(0.65, data[2][1], data[2][0]),linestyle ='--', ymin=0.0, ymax=0.95, alpha=0.3, color='darkgoldenrod')
# axs[1].axvline(x=np.interp(0.65, data[3][1], data[3][0]),linestyle ='--', ymin=0.0, ymax=0.95, alpha=0.3, color='navy')
# axs[1].axvline(x=np.interp(0.65, data[4][1], data[4][0]),linestyle ='--', ymin=0.0, ymax=0.95, alpha=0.3, color='forestgreen')

axs[1].grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs[1].grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs[1].minorticks_on()

# add a title for the entire figure
fig.suptitle('Straggler Convergence (Cora Full)', fontsize='xx-large')
axs[0].set_title('Loss Plot')
axs[1].set_title('Acc Plot')
axs[0].legend()
axs[1].legend()
# adjust spacing between subplots
fig.subplots_adjust(hspace=0.6, wspace=0.4)
plt.tight_layout()
# display the plot
plt.savefig("cora-full-straggler-effect.jpg", dpi=600)
plt.show()
