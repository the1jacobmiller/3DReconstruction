# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #


# Insert your package here
from skimage.color import rgb2xyz
from skimage.color import rgb2gray
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye
from scipy.sparse.linalg import lsqr as splsqr
import pdb
from utils import integrateFrankot, lRGB2XYZ
import numpy as np
import helper
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
Q3.2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # normalize points
    scale = 1./M
    T = np.diag([scale, scale, 1])
    pts1_norm = scale*pts1
    pts2_norm = scale*pts2

    # build A matrix
    N = pts1.shape[0]
    A = np.zeros((N,9))
    for i in range(N):
        x1 = pts1_norm[i,0]
        y1 = pts1_norm[i,1]
        x2 = pts2_norm[i,0]
        y2 = pts2_norm[i,1]
        A[i,:] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1.]

    # solve with SVD
    u,s,vh = np.linalg.svd(A, full_matrices=True)

    # Save e-vector corresponding to smallest e-value
    f = vh[-1,:]

    # Reshape to 3x3
    F_norm = f.reshape((3,3))

    # Enforce rank 2 constraint on F
    u,s,vh = np.linalg.svd(F_norm)
    s[-1] = 0 # set smallest singular value to 0
    F_norm = u@np.diag(s)@vh

    # Refine F
    F_norm = helper.refineF(F_norm, pts1_norm, pts2_norm)

    # Unnormalize F
    F = T.T@F_norm@T
    return F

def symbolicDet(f1,f2):
    tl1 = f1[4]
    tr1 = f1[5]
    bl1 = f1[7]
    br1 = f1[8]
    tl2 = f2[4]
    tr2 = f2[5]
    bl2 = f2[7]
    br2 = f2[8]
    a1 = tl1*br1 - bl1*tr1
    b1 = tl1*br2 + tl2*br1 - bl1*tr2 - bl2*tr1
    c1 = tl2*br2 - bl2*tr2

    tl1 = f1[3]
    tr1 = f1[5]
    bl1 = f1[6]
    br1 = f1[8]
    tl2 = f2[3]
    tr2 = f2[5]
    bl2 = f2[6]
    br2 = f2[8]
    a2 = tl1*br1 - bl1*tr1
    b2 = tl1*br2 + tl2*br1 - bl1*tr2 - bl2*tr1
    c2 = tl2*br2 - bl2*tr2

    tl1 = f1[3]
    tr1 = f1[4]
    bl1 = f1[6]
    br1 = f1[7]
    tl2 = f2[3]
    tr2 = f2[4]
    bl2 = f2[6]
    br2 = f2[7]
    a3 = tl1*br1 - bl1*tr1
    b3 = tl1*br2 + tl2*br1 - bl1*tr2 - bl2*tr1
    c3 = tl2*br2 - bl2*tr2

    v0 = f2[0]*c1 + f2[1]*c2 + f2[2]*c3
    v1 = f1[0]*c1 + f2[0]*b1 + f1[1]*c2 + f2[1]*b2 + f1[2]*c3 + f2[2]*b3
    v2 = f1[0]*b1 + f2[0]*a1 + f1[1]*b2 + f2[1]*a2 + f1[2]*b3 + f2[2]*a3
    v3 = f1[0]*a1 + f1[1]*a2 + f1[2]*a3
    return np.array([v0,v1,v2,v3])

'''
Q3.2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # choose 7 point correspondences
    rand_indices = np.random.choice(pts1.shape[0], 7, replace=False)
    test_pts1 = pts1[rand_indices,:]
    test_pts2 = pts2[rand_indices,:]

    # normalize points
    scale = 1./M
    T = np.diag([scale, scale, 1])
    pts1_norm = scale*test_pts1
    pts2_norm = scale*test_pts2

    # build A matrix
    A = np.zeros((7,9))
    for i in range(7):
        x1 = pts1_norm[i,0]
        y1 = pts1_norm[i,1]
        x2 = pts2_norm[i,0]
        y2 = pts2_norm[i,1]
        A[i,:] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1.]

    # solve with SVD
    u,s,vh = np.linalg.svd(A, full_matrices=True)

    # Save e-vector corresponding to smallest e-value
    f1 = vh[-1,:]
    f2 = vh[-2,:]

    vdet = symbolicDet(f1,f2)
    sol = np.roots(vdet)
    Fs = []
    for l in sol:
        if np.iscomplex(l):
            continue
        fsum = f1+l*f2
        F_norm = fsum.reshape((3,3))
        F = T.T@F_norm@T # Unnormalize F
        Fs.append(F)

    return Fs


'''
Q3.3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = K1.T@F@K2
    return E


'''
Q3.3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    N = pts1.shape[0]
    P_homogeneous = np.zeros((N,4))
    for i in range(N):
        # create A matrix
        A = np.zeros((4,4))

        # two equations for pt1
        p1 = C1[0,:]
        p2 = C1[1,:]
        p3 = C1[2,:]
        x = pts1[i,0]
        y = pts1[i,1]
        A[0,:] = y*p3.T-p2.T
        A[1,:] = p1.T-x*p3.T

        # two equations for pt2
        p1 = C2[0,:]
        p2 = C2[1,:]
        p3 = C2[2,:]
        x = pts2[i,0]
        y = pts2[i,1]
        A[2,:] = y*p3.T-p2.T
        A[3,:] = p1.T-x*p3.T

        # solve with SVD
        u,s,vh = np.linalg.svd(A, full_matrices=True)

        # Save e-vector corresponding to smallest e-value
        pt3D = vh[-1,:]
        P_homogeneous[i,:] = pt3D


    # convert points to inhomogeneous
    P_homogeneous[:,0] = P_homogeneous[:,0]/P_homogeneous[:,-1]
    P_homogeneous[:,1] = P_homogeneous[:,1]/P_homogeneous[:,-1]
    P_homogeneous[:,2] = P_homogeneous[:,2]/P_homogeneous[:,-1]
    P_homogeneous[:,3] = P_homogeneous[:,3]/P_homogeneous[:,-1]
    P = P_homogeneous[:,0:3]

    # reproject 3D points for camera 1 and 2
    x1hat = C1@P_homogeneous.T # 3xN matrix
    x1hat[0,:] = x1hat[0,:]/x1hat[-1,:]
    x1hat[1,:] = x1hat[1,:]/x1hat[-1,:]
    x1hat[2,:] = x1hat[2,:]/x1hat[-1,:]
    x1hat = x1hat[0:2,:].T # convert to inhomogeneous - Nx2 matrix

    x2hat = C2@P_homogeneous.T # 3xN matrix
    x2hat[0,:] = x2hat[0,:]/x2hat[-1,:]
    x2hat[1,:] = x2hat[1,:]/x2hat[-1,:]
    x2hat[2,:] = x2hat[2,:]/x2hat[-1,:]
    x2hat = x2hat[0:2,:].T # convert to inhomogeneous - Nx2 matrix

    # calculate reprojection error
    err = np.sum(np.linalg.norm(x1hat-pts1,axis=1)+np.linalg.norm(x2hat-pts2,axis=1))
    return P, err


'''
Q3.4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # convert images to grayscale
    im1g = rgb2gray(im1)
    im2g = rgb2gray(im2)

    # define window size and pad images
    window_size = 7
    pad_size = window_size//2
    im1_pad = np.pad(im1g,(pad_size,pad_size),
                    'constant', constant_values=(0, 0))
    im2_pad = np.pad(im2g,(pad_size,pad_size),
                    'constant', constant_values=(0, 0))

    # extract window of pixels around (x1,y1)
    im1_window = im1_pad[y1:y1+window_size,
                         x1:x1+window_size]

    # calculate epipolar line
    x_homogeneous = np.array([x1,y1,1.])
    line = F@x_homogeneous # [a,b,c] - ax+by+c=0

    # loop over every pixel on the epipolar line
    x2 = None
    y2 = None
    min_dist = None
    if abs(line[0]/line[1]) > 1: # slope greater than 45 deg.
        # loop over y
        # x = my+b  |  m=-b/a, b=-c/a
        m = -line[1]/line[0]
        b = -line[2]/line[0]
        start = max(y1-20,0)
        end = min(y1+20,im2.shape[0])
        for y in range(start,end):
            x = int(m*y+b)
            # check if x is in im2
            if x>0 and x<im2.shape[1]:
                # extract window of pixels around (x,y)
                test_window = im2_pad[y:y+window_size,
                                      x:x+window_size]
                dist = np.linalg.norm(im1_window-test_window)
                if min_dist == None or dist<min_dist:
                    x2 = x
                    y2 = y
                    min_dist = dist
    else: # slope less than 45 deg.
        # loop over x
        # y = mx+b  |  m=-a/b, b=-c/b
        m = -line[0]/line[1]
        b = -line[2]/line[1]
        start = max(x1-20,0)
        end = min(x1+20,im2.shape[1])
        for x in range(start,end):
            y = int(m*x+b)
            # check if y is in im2
            if y>0 and y<im2.shape[0]:
                # extract window of pixels around (x,y)
                test_window = im2_pad[y:y+window_size,
                                      x:x+window_size]
                dist = np.linalg.norm(im1_window-test_window)
                if min_dist == None or dist<min_dist:
                    x2 = x
                    y2 = y
                    min_dist = dist

    return x2,y2


'''
Q3.5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    N = pts1.shape[0]
    max_iter = 100
    tol = 0.0005

    # convert pts1 and pts2 to homogeneous
    pts1_homogeneous = np.hstack((pts1,np.ones((N,1))))
    pts2_homogeneous = np.hstack((pts2,np.ones((N,1))))

    bestF = None
    best_inliers = None
    max_inliers = -1
    for i in range(max_iter):
        # choose 8 point correspondences
        rand_indices = np.random.choice(N, 8, replace=False)
        test_pts1 = pts1[rand_indices,:]
        test_pts2 = pts2[rand_indices,:]

        # calculate F
        F = eightpoint(test_pts1, test_pts2, M)

        # calculate inliers
        inliers = np.zeros((N,),dtype='bool')
        inliers_count = 0
        for j in range(N):
            x = pts1_homogeneous[j,:]
            x_prime = pts2_homogeneous[j,:]
            err = abs(x_prime@F@x) # if F is perfect, x'Fx = 0
            if err < tol:
                inliers[j] = True
                inliers_count += 1

        # save if inliers > max_inliers
        if inliers_count > max_inliers:
            bestF = F
            best_inliers = inliers
            max_inliers = inliers_count
    print('Max inlier pct:', max_inliers/N)
    return bestF, best_inliers

def skew(v):
    v = v.reshape((3,))
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

'''
Q3.5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # source: https://www2.cs.duke.edu/courses/fall13/compsci527/notes/rodrigues.pdf
    theta = np.linalg.norm(r)
    if theta == 0:
        R = np.eye(3)
    else:
        u = r/theta
        R = np.eye(3)*np.cos(theta) + (1-np.cos(theta))*u@u.T + skew(u)*np.sin(theta)
    return R

def halfHemisphere(r):
    r = r.reshape((3,))
    if np.linalg.norm(r) == np.pi and \
        ((r[0]==0 and r[1]==0 and r[2]<0) or (r[0]==0 and r[1]<0) or (r[0]<0)):
        return -1*r
    else:
        return r

'''
Q3.5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # source: https://www2.cs.duke.edu/courses/fall13/compsci527/notes/rodrigues.pdf
    A = (R-R.T)/2.
    rho = np.array([A[2,1], A[0,2], A[1,0]])
    s = np.linalg.norm(rho)
    c = (R[0,0]+R[1,1]+R[2,2]-1)/2.

    print(s)
    print(c)

    r = np.zeros((3,))
    if s==0 and c==1:
        r = np.zeros((3,))
    elif s==0 and c==-1:
        v = (R+np.eye(3))[:,0]
        u = v/np.linalg.norm(v)
        r = halfHemisphere(u*np.pi)
    theta = np.arctan2(s,c)
    if np.sin(theta)!=0:
        u = rho/s
        r = u*theta
    return r


'''
Q3.5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original
            and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass


'''
Q3.5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    pass


def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Q4.1

    Render a sphere with a given center and radius. The camera is
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    m,n = res[0],res[1]
    image = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            x = (j-n//2)*pxSize
            y = (i-m//2)*pxSize
            if x**2+y**2>=rad**2:
                continue

            # calculate pseudo normal = (x,y,z)
            z = np.sqrt(rad**2-x**2-y**2)
            B = np.array([x,y,z])

            # calculate image intensity
            I = np.dot(light.T,B)
            image[i,j] = I
    plt.imshow(image,cmap='gray')
    plt.show()

    return image

def loadData(path = "../data/"):

    """
    Q4.2.1

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = []
    s = None
    for n in range(1,8):
        image_name = 'input_'+str(n)+'.tif'
        im = plt.imread(path+image_name)
        s = im.shape[:2]
        im_xyz = lRGB2XYZ(im) # convert RGB to XYZ
        im_y = im_xyz[:,:,1] # extract illuminance (Y) channel
        I.append(im_y.flatten())
    I = np.asarray(I)
    L = np.load('../data/sources.npy').T

    m,n = s
    assert I.shape == (7,m*n)
    assert L.shape == (3,7)

    return I,L,s

def estimatePseudonormalsCalibrated(I, L):

    """
    Q4.2.2

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = np.linalg.inv(L@L.T)@L@I
    assert B.shape == (3,I.shape[1])
    return B

def estimateAlbedosNormals(B):

    '''
    Q4.2.3

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B,axis=0)
    assert albedos.shape == (B.shape[1],)
    normals = B/albedos
    assert normals.shape == B.shape
    return albedos,normals

def displayAlbedosNormals(albedos, normals, s):

    """
    Q4.2.4

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    m,n = s
    albedoIm = albedos.reshape((m,n))
    normalIm = normals.T.reshape((m,n,3))

    # rescale normalIm
    normalIm = (normalIm+1)/2

    # display albedos
    plt.imshow(albedoIm,cmap='gray')
    plt.show()

    # display normals
    plt.imshow(normalIm,cmap='rainbow')
    plt.show()

    return albedoIm,normalIm

def estimateShape(normals, s):

    """
    Q4.3.1

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    zx = (-normals[0,:] / normals[2,:]).reshape(s)
    zy = (-normals[1,:] / normals[2,:]).reshape(s)

    z = integrateFrankot(zx, zy)
    return z


def plotSurface(surface):

    """
    Q4.3.1

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    m,n = surface.shape
    x, y = np.meshgrid(range(n), range(m), sparse=False, indexing='xy')

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(x,y,surface,cmap='coolwarm')
    plt.show()
