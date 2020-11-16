import matplotlib.pyplot as plt
import numpy as np
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = 640

# # 3.2.1
# print("--------- 3.2.1 ----------")
# F = sub.eightpoint(data['pts1'], data['pts2'], M)
# np.savez('q3_2_1.npz', F=F, M=M)
# print(F)
# helper.displayEpipolarF(im1, im2, F)

# # 3.2.2
# print("--------- 3.2.2 ----------")
# Fs = sub.sevenpoint(data['pts1'], data['pts2'], M)
# print('Found ', len(Fs), ' possible Fs')
# for F in Fs:
#     print(F)
#     helper.displayEpipolarF(im1, im2, F)

# 3.4.1
# print("--------- 3.4.1 ----------")
# F = sub.eightpoint(data['pts1'], data['pts2'], M)
# np.savez('q3_4_1.npz', F=F, pts1=data['pts1'], pts2=data['pts2'])
# helper.epipolarMatchGUI(im1, im2, F)

# 3.5.1
# print("--------- 3.5.1 ----------")
# noisy_data = np.load('../data/some_corresp_noisy.npz')
# F,inliers = sub.ransacF(data['pts1'], data['pts2'], M)
# helper.displayEpipolarF(im1, im2, F)

# # 3.5.2
# print("--------- 3.5.2 ----------")
# r = np.ones([3, 1])
# R = sub.rodrigues(r)
# r = sub.invRodrigues(R)
# print(r)

# # 4.1
# print("--------- 4.1 ----------")
# center = np.array([0,0,10]) # cm
# rad = 0.5 # cm
# pxSize = 0.0005 # 5 micro-meter
# res = np.array([3000,2500],dtype='int')
# light1 = np.array([1,1,1])/np.sqrt(3)
# light2 = np.array([1,-1,1])/np.sqrt(3)
# light3 = np.array([-1,-1,1])/np.sqrt(3)
# sphere1 = sub.renderNDotLSphere(center, rad, light1, pxSize, res)
# sphere2 = sub.renderNDotLSphere(center, rad, light2, pxSize, res)
# sphere3 = sub.renderNDotLSphere(center, rad, light3, pxSize, res)
#
# # render sphere with all 3 lighting sources
# sphere = (sphere1+sphere2+sphere3)/3.
# plt.imshow(sphere,cmap='gray')
# plt.show()

# 4.2.1
print("--------- 4.2.1 ----------")
I,L,s = sub.loadData()

# 4.2.2
print("--------- 4.2.2 ----------")
B = sub.estimatePseudonormalsCalibrated(I, L)

# 4.2.3
print("--------- 4.2.3 ----------")
albedos,normals = sub.estimateAlbedosNormals(B)

# # 4.2.4
# print("--------- 4.2.4 ----------")
# sub.displayAlbedosNormals(albedos, normals, s)

# 4.3.1
print("--------- 4.3.1 ----------")
surface = sub.estimateShape(normals, s)
sub.plotSurface(surface)
