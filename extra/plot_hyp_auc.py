import matplotlib.pyplot as plt
import numpy as np

# Generate multiple sets of example data for five figures
np.random.seed(0)
data_sets = []


data_sets.append({'ORE1': [0.7668, 0.7381, 0.9578],
                  'GO': [0.5009, 0.5435, 0.5716],
                  'FoodOn': [0.4108, 0.4548, 0.4727],
                  'PPI': [0.5731, 0.6083, 0.6891],
                  'X:Embedding Size': ['50', '100', '200']
                  })

data_sets.append({'ORE1': [0.9552, 0.9578, 0.9578],
                  'GO': [0.5610, 0.5716, 0.5779],
                  'FoodOn': [0.4581, 0.4727, 0.4502],
                  'PPI': [0.6891, 0.6680, 0],
                  'X:Number of negatives': ['1', '2', '4']
                  })

data_sets.append({'ORE1': [0.9883, 0.9876, 0.9953],
                  'GO': [0.9149, 0.9247, 0.9333],
                  'FoodOn': [0.8686, 0.8833, 0.9021],
                  'PPI': [0.9559, 0.9598, 0.963],
                  'X:Embedding Size': ['50', '100', '200']
                  })

data_sets.append({'ORE1': [0.9952, 0.9953, 0.9953],
                  'GO': [0.9313, 0.9333, 0.9330],
                  'FoodOn': [0.8994, 0.9021, 0.8981],
                  'PPI': [0.9630, 0.9638, 0],
                  'X:Number of negatives': ['1', '2', '4']
                  })
 
# Plotting with a 2-2-1 layout and ensuring the last one has the same dimensions
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

# Flatten the 2D array of axes for easy iteration
axs = axs.flatten()

# Plot the first four figures
for i in range(4):
    x_label = list(data_sets[i].keys())[-1]
    x_label_trimmed = x_label[2:]

    for key in ['ORE1', 'GO', 'FoodOn', 'PPI']:
        y = data_sets[i][key]
        x = data_sets[i][x_label]

        filtered_y = [i for i in y if i != 0]
        filtered_x = [x[i] for i in range(len(x)) if y[i] != 0]

        axs[i].plot(filtered_x, filtered_y, label=key, linewidth=3, marker='o', markersize=10)
    
        
    axs[i].set_xlabel(x_label_trimmed, fontsize=30)
    if i < 2:
        axs[i].set_ylabel('Hits@100', fontsize=30)
    else:
        axs[i].set_ylabel('ROC AUC', fontsize=30)
    # axs[i].set_title(f'Figure {i + 1}')
    axs[i].legend(loc='upper left', fontsize=20)
    axs[i].tick_params(axis='both', which='major', labelsize=20)
    axs[i].grid(True)

# Hide the last subplot (6th subplot)

# Create a new axis for the last figure

# Adjust layout to make sure all plots are properly aligned
plt.tight_layout()
# plt.show()
#save as pdf with 300 dpi
plt.savefig('/home/zhapacfp/Latex/cate_nesy/hyp_simple.pdf', dpi=300)
