import matplotlib.pyplot as plt
import numpy as np

with open('./precisions.txt', 'r') as file:
    data = file.readlines()
    data = list(map(lambda x: round(float(x.rstrip()), 4), data))

    plt.bar(range(1, 51), data, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Precisions Of Queries')
    plt.xlabel('Queries')
    plt.ylabel('Precision')

    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks([1, 10, 20, 30, 40, 50])

    plt.savefig('./bar_final.png')
    plt.close()

# plt.bar(range(1, 51), precision_values,
#         color='blue', alpha=0.7, edgecolor='black')

# plt.title('Precision Values for Queries')
# plt.xlabel('Queries')
# plt.ylabel('Precision')

# plt.savefig('./bar3.png')

# plt.close()
