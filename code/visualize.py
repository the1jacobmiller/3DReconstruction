# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import submission as sub
from findM2 import test_M2_solution

'''
Q3.4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

# load images
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

# load intrinsic matrices
intrinsics = np.load('../data/intrinsics.npz')

# load points from templeCoords
data = np.load('../data/templeCoords.npz')
x1 = data['x1']
y1 = data['y1']
N = len(x1)

# calculate F
M = 640
data = np.load('../data/some_corresp.npz')
F = sub.eightpoint(data['pts1'], data['pts2'], M)

# find point correspondences - pts1 and pts2
pts1 = np.zeros((N,2),dtype='int')
pts2 = np.zeros((N,2),dtype='int')
pts1[:,0] = x1[:,0]
pts1[:,1] = y1[:,0]

for i in range(N):
    x = pts1[i,0]
    y = pts1[i,1]
    x2, y2 = sub.epipolarCorrespondence(im1, im2, F, x, y)
    pts2[i,0] = x2
    pts2[i,1] = y2

# find 3D points from point correspondences
M2, C2, P = test_M2_solution(pts1, pts2, intrinsics)

# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

# Creating plot
ax.scatter3D(P[:,1], P[:,0], P[:,2], color = "blue")
plt.title("3D reconstruction plot")

# show plot
plt.show()

# save F, M1, M2, C1, C2
K1 = intrinsics['K1']
M1 = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0]])
C1 = K1@M1
np.savez('q3_4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)
