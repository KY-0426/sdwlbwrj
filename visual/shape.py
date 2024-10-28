import numpy as np

x = np.load('../data/train_joint.npy',mmap_mode='r')
print(x.shape)

x = np.load('../data/test_A_bone.npy',mmap_mode='r')
print(x.shape)
print(x.max())

#x = np.load('data/test_joint_B.npy',mmap_mode='r')
#print(x.shape)