# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #

import helper
import submission as sub
import numpy as np

'''
Q3.3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_5.npz
'''


def test_M2_solution(pts1, pts2, intrinsics):
    '''
    Estimate all possible M2 and return the correct M2 and 3D points P
    :param pred_pts1:
    :param pred_pts2:
    :param intrinsics:
    :return: M2, the extrinsics of camera 2
    		 C2, the 3x4 camera matrix
    		 P, 3D points after triangulation (Nx3)
    '''

    # load intrinsic matrices
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']

    # calculate C1
    M1 = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0]])
    C1 = K1@M1

    # calculate F and E
    M = 640
    F = sub.eightpoint(pts1, pts2, M)
    E = sub.essentialMatrix(F, K1, K2)
    print(E)

    # test potential M2s
    M2s = helper.camera2(E)
    for i in range(M2s.shape[2]):
        # calculate C2
        M2 = M2s[:,:,i]
        R = M2[:,0:3]

        # test det(R)
        if round(np.linalg.det(R)) != 1:
            continue

        # test for positive z from triangulation
        C2 = K2@M2
        P,err = sub.triangulate(C1, pts1, C2, pts2)
        if (P[:,2] > 0).all():
            print('Reprojection error:',err)
            return M2, C2, P
    return None, None, None


if __name__ == '__main__':
    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    intrinsics = np.load('../data/intrinsics.npz')

    M2, C2, P = test_M2_solution(pts1, pts2, intrinsics)
    np.savez('q3_3_3', M2=M2, C2=C2, P=P)
