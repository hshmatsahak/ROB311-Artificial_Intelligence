# Importing the Numpy Package
import numpy as np

# Numpy Array
scores = np.array([100,67,92,87,66,89,76,22])

# Getting indices of N = 3 maximum values
x = np.argsort(scores)[::-1][:3]
print("Indices:",x)

# Getting N maximum values
print("Values:",scores[x])