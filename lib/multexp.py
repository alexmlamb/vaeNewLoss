
import numpy as np

x1 = np.random.uniform(size = (100,3,32,32)).transpose(0,2,3,1)

'''
Want to map this to:

    (100,96,32,32)
'''

W = np.random.uniform(size = (3,96))

print np.dot(x1, W).transpose(0,3,1,2).shape



