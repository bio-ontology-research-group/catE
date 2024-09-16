import matplotlib.pyplot as plt
import numpy as np

# Generate multiple sets of example data for five figures
np.random.seed(0)
data_sets = []

data_sets.append({'ORE1': [0.9952, 0.9953, 0.9953],
                  'GO': [0.9313, 0.9333, 0.9330],
                  'FoodOn': [0.8994, 0.9021, 0.8981],
                  'PPI': [0.9630, 0.9638, 0],
                  'X:Number of negatives': ['1', '2', '4']
                  })

data_sets.append({'ORE1': [0, 0.9876, 0.9953],
                  'GO': [0.9149, 0.9247, 0.9333],
                  'FoodOn': [0.8686, 0.8833, 0.9021],
                  'PPI': [0.9559, 0.9598, 0.963],
                  'X:Embedding Size': ['50', '100', '200']
                  })

for _ in range(3):
    data_sets.append({
        'ORE1': np.random.rand(3),
        'GO': np.random.rand(3),
        'FoodOn': np.random.rand(3),
        'PPI': np.random.rand(3),
        'X:metric': ['1', '2', '4']
    })

# Plotting with a 2-2-1 layout and ensuring the last one has the same dimensions
fig, axs = plt.subplots(3, 2, figsize=(14, 18))

# Flatten the 2D array of axes for easy iteration
axs = axs.flatten()

# Plot the first four figures
for i in range(4):
    x_label = list(data_sets[i].keys())[-1]
    x_label_trimmed = x_label[2:]
    axs[i].plot(data_sets[i][x_label], data_sets[i]['ORE1'], label='ORE1')
    axs[i].plot(data_sets[i][x_label], data_sets[i]['GO'], label='GO')
    axs[i].plot(data_sets[i][x_label], data_sets[i]['FoodOn'], label='FoodOn')
    axs[i].plot(data_sets[i][x_label], data_sets[i]['PPI'], label='PPI')

    axs[i].set_xlabel(x_label_trimmed)
    axs[i].set_ylabel('Values')
    axs[i].set_title(f'Figure {i + 1}')
    axs[i].legend()
    axs[i].grid(True)

# Hide the last subplot (6th subplot)
fig.delaxes(axs[-1])

# Create a new axis for the last figure
ax_last = fig.add_subplot(3, 2, 5)

ax_last.plot(data_sets[4]['X:metric'], data_sets[4]['ORE1'], label='ORE1')
ax_last.plot(data_sets[4]['X:metric'], data_sets[4]['GO'], label='GO')
ax_last.plot(data_sets[4]['X:metric'], data_sets[4]['FoodOn'], label='FoodOn')
ax_last.plot(data_sets[4]['X:metric'], data_sets[4]['PPI'], label='PPI')

ax_last.set_xlabel('E')
ax_last.set_ylabel('Values')
ax_last.set_title('Figure 5')
ax_last.legend()
ax_last.grid(True)

# Adjust layout to make sure all plots are properly aligned
plt.tight_layout()
plt.show()
