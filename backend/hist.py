import matplotlib.pyplot as plt
import numpy as np

data1 = []
data2 = []

with open('precisions.txt', 'r') as file:
    data = list(map(lambda x: x.rstrip().split(), file.readlines()))
    data = list(map(lambda x: (float(x[0]), float(x[1])), data))
    data1 = [d1 for (d1, _) in data]
    data2 = [d2 for (_, d2) in data]
    print(data1)

# with open('reinforce_precisions.txt', 'r') as file:
#     data = list(map(lambda x: x.rstrip(), file.readlines()))
#     data = list(map(lambda x: float(x), data))
#     data2 = data
#     print(data)

# Example data
# categories = ['Category A', 'Category B', 'Category C']


# Set the width of the bars
bar_width = 0.40

# Create an array of indices to position the bars
ind = np.arange(len(data1[:15]))

# Plotting the side-by-side bars
plt.bar(ind, data1[:15], width=bar_width,
        label='Normal (td-idf)', color='skyblue')
plt.bar(ind + bar_width, data2[:15], width=bar_width,
        label='Relevance Feedback', color='orange')

# Add labels, title, and legend
plt.xlabel('Queries')
plt.ylabel('Precision')
plt.title(
    'Comparison of precision of normal (tf-idf) and relevance feedback algorithm')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# plt.xticks(ind + bar_width / 2, categories)
plt.legend()

# Show the plot
plt.savefig('./charts/normal_vs_rf_v2.png')
plt.close()
