import matplotlib.pyplot as plt
import numpy as np
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('axes', labelsize='xx-large', titlesize='xx-large')
plt.rcParams['legend.title_fontsize'] = 'xx-large'
plt.rcParams['legend.fontsize'] = 'medium'

# data = [
#
#
# [[45.56, 96.88, 146.98, 197.21, 248.3, 298.84, 350.59, 401.76, 453.54, 503.78, 553.41, 603.66],
#  [0.4198, 0.5675, 0.6567, 0.7085, 0.7587, 0.8078, 0.8271, 0.8564, 0.8717, 0.8829, 0.8925, 0.8998],
#  [2.8956, 2.3803, 1.9918, 1.698, 1.4658, 1.2991, 1.1653, 1.062, 0.989, 0.9341, 0.8895, 0.8513]],
# [[45.07, 95.1, 145.6, 195.62, 246.53, 296.56, 346.7, 395.91, 444.92, 494.34, 544.22, 592.92, 642.23],
#  [0.4524, 0.5212, 0.584, 0.7337, 0.7693, 0.7841, 0.8073, 0.8255, 0.8442, 0.8587, 0.8734, 0.8912, 0.8988],
#  [2.8297, 2.3017, 1.9588, 1.6934, 1.4797, 1.3326, 1.2177, 1.1286, 1.0589, 0.9978, 0.9489, 0.9095, 0.8757]]
# ]
# # create a figure with a 2x2 grid of subplots
# fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))
#
# axs.plot(data[0][0], data[0][1], 'magenta', linewidth=1.5, label="N = 8")
# axs.plot(data[1][0], data[1][1], 'crimson', linewidth=1.5, label="N = 16")
# axs.plot(data[2][0], data[2][1], 'darkgoldenrod', linewidth=1.5, label="N = 24")
# axs.plot(data[3][0], data[3][1], 'navy', linewidth=1.5, label="N = 32")
#
# axs.axvline(x=np.interp(0.90, data[0][1], data[0][0]),linestyle ='--', ymin=0.0, ymax=0.6, alpha=0.3, color='magenta')
# axs.axvline(x=np.interp(0.90, data[1][1], data[1][0]),linestyle ='--', ymin=0.0, ymax=0.6, alpha=0.3, color='crimson')
# axs.axvline(x=np.interp(0.90, data[2][1], data[2][0]),linestyle ='--', ymin=0.0, ymax=0.6, alpha=0.3, color='darkgoldenrod')
# axs.axvline(x=np.interp(0.90, data[3][1], data[3][0]),linestyle ='--', ymin=0.0, ymax=0.6, alpha=0.3, color='navy')
#
#
# axs.set_xlabel('Wall Time (s)')
# axs.set_ylabel('Test Accuracy')
# # axs.set_ylim([0.9, 1])
# axs.axhline(y = 0.9, color = 'k', linestyle = '--', linewidth=1.5)
# axs.grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
# axs.grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
# axs.minorticks_on()
#
# # add a title for the entire figure
# fig.suptitle('Scalability Plot (mw-PR on Reddit)\nfor P = 3', fontsize='xx-large')
# # axs.set_title('Acc Plot')
# axs.legend()
# # adjust spacing between subplots
# plt.tight_layout()
# # display the plot
# plt.savefig("p-red-scalability.jpg", dpi=600)
# plt.show()

# ----------------------------------------------------------------------------

# data = [
# [[46.19, 96.66, 146.82, 197.24, 247.91, 297.61, 346.85], [0.6511, 0.7415, 0.8126, 0.8665, 0.8916, 0.9046, 0.912], [1.9912, 1.5049, 1.1956, 0.9963, 0.8664, 0.7771, 0.7121]],
# [[45.97, 97.01, 148.82, 199.27, 250.19, 300.52, 351.57, 401.9, 452.38, 502.53], [0.1478, 0.4555, 0.6599, 0.7521, 0.7908, 0.8341, 0.8652, 0.8867, 0.9015, 0.9115], [3.5286, 2.6664, 1.9741, 1.5592, 1.2953, 1.1183, 0.989, 0.8936, 0.822, 0.7662]],
# [[46.46, 98.15, 150.21, 201.09, 252.86, 303.75, 355.84, 407.43, 459.91, 512.15, 563.77, 614.78, 665.5], [0.409, 0.53, 0.6271, 0.7084, 0.7325, 0.7599, 0.7984, 0.8228, 0.8502, 0.8675, 0.8788, 0.889, 0.8966], [3.1999, 2.6831, 2.2635, 1.9208, 1.6612, 1.4588, 1.3081, 1.1905, 1.0983, 1.0244, 0.9636, 0.9144, 0.8719]]
# ]

data = [
[[57.91, 129.05, 200.01, 270.93, 342.05], [0.0009, 0.6249, 0.6901, 0.748, 0.8068], [3.8445, 2.6973, 2.0715, 1.6503, 1.3605]],
[[50.51, 121.69, 192.03, 262.31], [0.0267, 0.6108, 0.6961, 0.8015], [3.8701, 2.483, 1.8525, 1.4415]],
[[48.0, 117.46, 188.43], [0.533, 0.6781, 0.7894], [3.0215, 2.0724, 1.5111]],
[[47.47, 117.98, 188.91], [0.6616, 0.7898, 0.8477], [2.0721, 1.5085, 1.1841]]
]
# create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

axs.plot(data[0][0], data[0][1], 'magenta', linewidth=1.5, label="N = 8")
axs.plot(data[1][0], data[1][1], 'crimson', linewidth=1.5, label="N = 12")
axs.plot(data[2][0], data[2][1], 'darkgoldenrod', linewidth=1.5, label="N = 16")
axs.plot(data[3][0], data[3][1], 'navy', linewidth=1.5, label="N = 20")

axs.axvline(x=np.interp(0.75, data[0][1], data[0][0]),linestyle ='--', ymin=0.0, ymax=0.87, alpha=0.3, color='magenta')
axs.axvline(x=np.interp(0.75, data[1][1], data[1][0]),linestyle ='--', ymin=0.0, ymax=0.87, alpha=0.3, color='crimson')
axs.axvline(x=np.interp(0.75, data[2][1], data[2][0]),linestyle ='--', ymin=0.0, ymax=0.87, alpha=0.3, color='darkgoldenrod')
axs.axvline(x=np.interp(0.75, data[3][1], data[3][0]),linestyle ='--', ymin=0.0, ymax=0.87, alpha=0.3, color='navy')

axs.set_xlabel('Wall Time (s)')
axs.set_ylabel('Test Accuracy')
# axs.set_ylim([0.85, 1])
axs.axhline(y = 0.75, color = 'k', linestyle = '--', linewidth=1.5)
axs.grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs.grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs.minorticks_on()

# add a title for the entire figure
fig.suptitle('Scalability Plot (g-ARAR on Reddit)', fontsize='xx-large')
# axs.set_title('Acc Plot')
axs.legend()
# adjust spacing between subplots
plt.tight_layout()
# display the plot
plt.savefig("g-ar-ar-scalability-good.jpg", dpi=600)
plt.show()


# # ----------------------------------------------------------------------------

# data = [
# [[45.31, 99.13, 149.95, 200.38, 254.89, 305.65, 355.5, 405.55, 455.93], [0.5045, 0.6342, 0.7159, 0.7583, 0.8109, 0.8455, 0.8705, 0.8885, 0.9008], [3.0301, 2.1241, 1.7021, 1.4255, 1.206, 1.0453, 0.9332, 0.8492, 0.7866]],
# [[45.82, 96.63, 153.67, 207.41, 258.8, 311.18, 368.29, 419.44, 469.99, 521.5, 571.33, 621.67], [0.4791, 0.5505, 0.6775, 0.745, 0.7749, 0.8052, 0.8283, 0.8508, 0.8674, 0.8794, 0.8921, 0.9007], [3.1925, 2.6553, 1.9041, 1.6267, 1.4687, 1.2901, 1.1496, 1.0572, 0.9881, 0.931, 0.882, 0.8399]],
# [[45.56, 107.54, 163.96, 218.73, 272.72, 326.8, 380.95, 433.85, 486.4, 538.91, 595.47, 648.57, 701.76], [0.1036, 0.4395, 0.4899, 0.494, 0.596, 0.6545, 0.6776, 0.7002, 0.7226, 0.7438, 0.7688, 0.78, 0.7899], [3.5445, 2.9498, 2.641, 2.4188, 2.2242, 2.0593, 1.918, 1.7952, 1.7011, 1.6186, 1.5386, 1.4708, 1.4066]]
# ]

data = [
[[59.65, 209.93, 358.63, 508.36, 658.35, 807.83], [0.4879, 0.541, 0.629, 0.6719, 0.7048, 0.7497], [2.8329, 2.4911, 2.2128, 1.9681, 1.7683, 1.6014]],
[[46.32, 117.24, 188.73, 258.5, 328.78], [0.5419, 0.5793, 0.6769, 0.7382, 0.7911], [2.8016, 2.3454, 1.9121, 1.6967, 1.5425]],
[[47.04, 117.59, 188.32], [0.5617, 0.6938, 0.7889], [2.6582, 1.8339, 1.4595]],
[[46.86, 119.96, 217.34, 289.2], [0.629, 0.6466, 0.7426, 0.7912], [2.3379, 1.9247, 1.479, 1.4129]]
]
# create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

axs.plot(data[0][0], data[0][1], 'magenta', linewidth=1.5, label="N = 8")
axs.plot(data[1][0], data[1][1], 'crimson', linewidth=1.5, label="N = 12")
axs.plot(data[2][0], data[2][1], 'darkgoldenrod', linewidth=1.5, label="N = 16")
axs.plot(data[3][0], data[3][1], 'navy', linewidth=1.5, label="N = 20")



axs.axvline(x=np.interp(0.75, data[0][1], data[0][0]),linestyle ='--', ymin=0.0, ymax=0.82, alpha=0.3, color='magenta')
axs.axvline(x=np.interp(0.75, data[1][1], data[1][0]),linestyle ='--', ymin=0.0, ymax=0.82, alpha=0.3, color='crimson')
axs.axvline(x=np.interp(0.75, data[2][1], data[2][0]),linestyle ='--', ymin=0.0, ymax=0.82, alpha=0.3, color='darkgoldenrod')
axs.axvline(x=np.interp(0.75, data[3][1], data[3][0]),linestyle ='--', ymin=0.0, ymax=0.82, alpha=0.3, color='navy')

axs.set_xlabel('Wall Time (s)')
axs.set_ylabel('Test Accuracy')
# axs.set_ylim([0.85, 1])
axs.axhline(y = 0.75, color = 'k', linestyle = '--', linewidth=1.5)
axs.grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs.grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs.minorticks_on()

# add a title for the entire figure
fig.suptitle('Scalability Plot (g-ARPS on Reddit)', fontsize='xx-large')
# axs.set_title('Acc Plot')
axs.legend()
# adjust spacing between subplots
plt.tight_layout()
# display the plot
plt.savefig("g-ar-ps-scalability-good.jpg", dpi=600)
plt.show()