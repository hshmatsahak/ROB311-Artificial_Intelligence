import numpy as np

M = np.ones([5,5,2])
utils = np.ones(5)
print(M[1, :, :])
print(np.matmul(utils, M[1,:,:]))
print(utils)

test2 = M[1,:,1]
print(np.dot(test2, utils))

# while (True):
#     print(np.random.uniform(0,1))

print(np.amax(np.array([3,4,5,6,3])))