import matplotlib.pyplot as plt
import numpy as np
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('axes', labelsize='xx-large', titlesize='xx-large')
plt.rcParams['legend.title_fontsize'] = 'xx-large'
plt.rcParams['legend.fontsize'] = 'medium'

data = [
]
# create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))

axs.plot(data[0][0], data[0][1], 'magenta', linewidth=1.5, label="N = 8")
axs.plot(data[1][0], data[1][1], 'crimson', linewidth=1.5, label="N = 16")
axs.plot(data[2][0], data[2][1], 'darkgoldenrod', linewidth=1.5, label="N = 24")
axs.plot(data[3][0], data[3][1], 'navy', linewidth=1.5, label="N = 32")

axs.axvline(x=np.interp(0.96, data[0][1], data[0][0]),linestyle ='--', ymin=0.0, ymax=0.6, alpha=0.3, color='magenta')
axs.axvline(x=np.interp(0.96, data[1][1], data[1][0]),linestyle ='--', ymin=0.0, ymax=0.6, alpha=0.3, color='crimson')
axs.axvline(x=np.interp(0.96, data[2][1], data[2][0]),linestyle ='--', ymin=0.0, ymax=0.6, alpha=0.3, color='darkgoldenrod')
axs.axvline(x=np.interp(0.96, data[3][1], data[3][0]),linestyle ='--', ymin=0.0, ymax=0.6, alpha=0.3, color='navy')


axs.set_xlabel('Wall Time (s)')
axs.set_ylabel('Test Accuracy')
# axs.set_ylim([0.9, 1])
axs.axhline(y = 0.9, color = 'k', linestyle = '--', linewidth=1.5)
axs.grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
axs.grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
axs.minorticks_on()

# add a title for the entire figure
fig.suptitle('Scalability Plot (mw-PR on Reddit)\nfor P = 3', fontsize='xx-large')
# axs.set_title('Acc Plot')
axs.legend()
# adjust spacing between subplots
plt.tight_layout()
# display the plot
plt.savefig("p-red-scalability.jpg", dpi=600)
plt.show()

# ----------------------------------------------------------------------------

data = [
[[63.98, 127.5, 191.06, 254.69, 318.19, 381.85], [0.8908, 0.9238, 0.9419, 0.9521, 0.9605, 0.9659], [112.1179, 24.8905, 17.4468, 13.2639, 11.7257, 10.3627]],
[[58.96, 117.59, 176.23, 234.88, 293.61], [0.9014, 0.9392, 0.9542, 0.9643, 0.9702], [111.7097, 23.0268, 14.6148, 11.13, 9.6491]],
[[59.09, 117.75, 176.57, 235.05, 292.28], [0.9136, 0.9479, 0.9624, 0.9705, 0.9754], [113.0153, 22.7604, 14.1393, 10.7756, 8.7307]],
[[58.28, 121.25, 184.42, 242.79], [0.9154, 0.9515, 0.9657, 0.9733], [112.4064, 21.2462, 12.7405, 10.0035]],
]
# create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

axs.plot(data[0][0], data[0][1], 'magenta', linewidth=1.5, label="N = 8")
axs.plot(data[1][0], data[1][1], 'crimson', linewidth=1.5, label="N = 16")
axs.plot(data[2][0], data[2][1], 'darkgoldenrod', linewidth=1.5, label="N = 24")
axs.plot(data[3][0], data[3][1], 'navy', linewidth=1.5, label="N = 32")

axs.axvline(x=np.interp(0.965, data[0][1], data[0][0]),linestyle ='--', ymin=0.0, ymax=0.75, alpha=0.3, color='magenta')
axs.axvline(x=np.interp(0.965, data[1][1], data[1][0]),linestyle ='--', ymin=0.0, ymax=0.75, alpha=0.3, color='crimson')
axs.axvline(x=np.interp(0.965, data[2][1], data[2][0]),linestyle ='--', ymin=0.0, ymax=0.75, alpha=0.3, color='darkgoldenrod')
axs.axvline(x=np.interp(0.965, data[3][1], data[3][0]),linestyle ='--', ymin=0.0, ymax=0.75, alpha=0.3, color='navy')

axs.set_xlabel('Wall Time (s)')
axs.set_ylabel('Test Accuracy')
# axs.set_ylim([0.85, 1])
axs.axhline(y = 0.9, color = 'k', linestyle = '--', linewidth=1.5)
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
plt.savefig("g-ar-ar-scalability.jpg", dpi=600)
plt.show()


# # ----------------------------------------------------------------------------

data = [
[[64.39, 128.25, 192.15, 255.79, 319.27, 382.81, 446.5, 510.24], [0.8908, 0.9172, 0.9344, 0.9463, 0.9548, 0.9607, 0.9651, 0.9688], [112.1179, 24.3597, 18.5008, 14.2412, 12.9928, 11.307, 10.0156, 9.2227]],
[[59.29, 118.3, 177.14, 236.87, 295.92], [0.9014, 0.9334, 0.9497, 0.9598, 0.9662], [111.7097, 22.4734, 15.4793, 11.9064, 10.5576]],
[[58.5, 116.77, 176.04, 236.23, 296.51], [0.9095, 0.9416, 0.9528, 0.9624, 0.9724], [113.0153, 23.2071, 15.4591, 12.3937, 10.2458]],
[[58.36, 116.47, 174.13, 232.19], [0.9144, 0.9434, 0.9592, 0.9707], [112.4064, 21.4213, 14.5256, 11.201]]
]
# create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

axs.plot(data[0][0], data[0][1], 'magenta', linewidth=1.5, label="N = 8")
axs.plot(data[1][0], data[1][1], 'crimson', linewidth=1.5, label="N = 16")
axs.plot(data[2][0], data[2][1], 'darkgoldenrod', linewidth=1.5, label="N = 24")
axs.plot(data[3][0], data[3][1], 'navy', linewidth=1.5, label="N = 32")



axs.axvline(x=np.interp(0.965, data[0][1], data[0][0]),linestyle ='--', ymin=0.0, ymax=0.75, alpha=0.3, color='magenta')
axs.axvline(x=np.interp(0.965, data[1][1], data[1][0]),linestyle ='--', ymin=0.0, ymax=0.75, alpha=0.3, color='crimson')
axs.axvline(x=np.interp(0.965, data[2][1], data[2][0]),linestyle ='--', ymin=0.0, ymax=0.75, alpha=0.3, color='darkgoldenrod')
axs.axvline(x=np.interp(0.965, data[3][1], data[3][0]),linestyle ='--', ymin=0.0, ymax=0.75, alpha=0.3, color='navy')

axs.set_xlabel('Wall Time (s)')
axs.set_ylabel('Test Accuracy')
# axs.set_ylim([0.85, 1])
axs.axhline(y = 0.9, color = 'k', linestyle = '--', linewidth=1.5)
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
plt.savefig("g-ar-ps-scalability.jpg", dpi=600)
plt.show()