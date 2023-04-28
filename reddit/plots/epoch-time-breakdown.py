from matplotlib import pyplot as plt
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('axes', labelsize='xx-large', titlesize='xx-large')
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'medium'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def plot_clustered_stacked(dfall, labels=None, title="Avg Epoch Time Breakdown (Reddit)",  H="o", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall :
        axe = df.plot(kind="bar",
                      linewidth=0.5,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      edgecolor = 'black',
                      **kwargs)

    h,l = axe.get_legend_handles_labels()
    for i in range(0, n_df * n_col, n_col):
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches:
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    axe.set_ylabel("Avg. Epoch Time (sec)")
    axe.set_xlabel("Proposed Methods")
    axe.grid(True, which='major', linestyle='-', color='black', axis='y', alpha=0.5)
    axe.grid(True, which='minor', linestyle='--', color='red', alpha=0.2, axis='y')
    axe.minorticks_on()
    axe.set_axisbelow(True)

    n=[]
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5], title="Time")
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1])
    axe.add_artist(l1)
    return axe

list1 = [
    [29.706, 4.885, 0],
    [30.080, 0.295, 0],
    [29.667, 0.476, 0],
    [28.921, 1.176, 0.919]
]
df1 = pd.DataFrame(np.array(list1),
                   index=['s-AR', 'g-ARAR', 'g-ARPS', 'mw-PR'],
                   columns=["Comp", "Comm", "Waiting"])

list2 = [
    [19.018, 4.885, 10.689],
    [17.858, 0.295, 2.098],
    [17.751, 0.476, 0.045],
    [17.158, 1.176, 0.394]
]
df2 = pd.DataFrame(np.array(list2),
                   index=['s-AR', 'g-ARAR', 'g-ARPS', 'mw-PR'],
                   columns=["Comp", "Comm", "Waiting"])

list3 = [
    [5.985, 4.885, 23.722],
    [5.968, 0.295, 1.331],
    [5.902, 0.476, 0],
    [5.258, 1.176, 0.392]
]
df3 = pd.DataFrame(np.array(list3),
                   index=['s-AR', 'g-ARAR', 'g-ARPS', 'mw-PR'],
                   columns=["Comp", "Comm", "Waiting"])

plot_clustered_stacked([df1, df2, df3],["slow", "medium", "fast"])
plt.tight_layout()
plt.savefig("reddit-epoch-time-breakdown.jpg", dpi=600)
plt.show()