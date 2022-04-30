from matplotlib.pyplot import cla
import numpy as np
examples = np.array([[1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
                    [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
                    [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
                    [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
                    [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
                    [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
                    [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                    [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1]])
print(examples.shape)
class_label = examples[:, -1]
unique, counts = np.unique(class_label, return_counts=True)
frac_counts = np.divide(counts, np.sum(counts))
print(np.sum(counts))
print(counts)
print(frac_counts)
print(unique)
subexamples = examples[examples[:,-1] == 98]
print(subexamples)

x = np.array([3,4,5,6,7,8])
y = x[[0,2,4]]
print(y)