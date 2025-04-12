import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
fruit_weights = [
    np.random.normal(130, 10, size=100),
    np.random.normal(125, 20, size=100),
    np.random.normal(120, 30, size=100),
]
labels = ['Black', 'White', 'Asian', 'Latino', 'American Indian', 'Native Hawaiian']

fig, ax = plt.subplots()

bplot = ax.boxplot(fruit_weights,
                   patch_artist=True,  # fill with color
                   tick_labels=labels)  # will be used to label x-ticks

# # fill with colors
# for patch in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

plt.savefig('./test_image.png')