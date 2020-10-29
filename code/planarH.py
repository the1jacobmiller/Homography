import numpy as np
import cv2
from BRIEF import briefLite, briefMatch
import copy


def computeH(p1, p2):
    """
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
            equation
    """
    assert p1.shape[1] == p2.shape[1]
    assert p1.shape[0] == 2

    N = p1.shape[1]
    A = np.zeros((2*N,9))
    # Compute A
    for i in range(N):
        x = p1[0,i]
        y = p1[1,i]
        u = p2[0,i]
        v = p2[1,i]
        A_i = np.array([[-x, -y, -1, 0, 0, 0, x*u, y*u, u],
                    [0, 0, 0, -x, -y, -1, x*v, y*v, v]])
        A[2*i:2*i+2,:] = A_i

    # Compute SVD of A
    u, s, vh = np.linalg.svd(A, full_matrices=True)

    # Save e-vector corresponding to smallest e-value
    h = vh[-1,:]

    # Reshape to 3x3
    H2to1 = h.reshape((3,3))

    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    """
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    """

    max_inliers = 0
    bestH = np.zeros((3,3))

    for iter in range(num_iter):
        # randomly select 4 correspondences
        random_ind = np.random.choice(matches.shape[0],4, replace=False)
        random_matches = matches[random_ind]

        # compute H
        p1 = locs1[random_matches[:,0]].T[0:2,:]
        p2 = locs2[random_matches[:,1]].T[0:2,:]
        H2to1 = computeH(p1, p2)

        # count inliers
        p1s = locs1[matches[:,0]].T[0:2,:]
        p2s = locs2[matches[:,1]].T[0:2,:]
        p1_homogeneous = np.ones((3,matches.shape[0]))
        p2_homogeneous = np.ones((3,matches.shape[0]))
        p1_homogeneous[0:2,:] = p1s
        p2_homogeneous[0:2,:] = p2s

        p2hat = H2to1.dot(p1_homogeneous)
        p2hat = (p2hat / p2hat[2,:])[0:2,:]
        e_matrix = np.abs(p2hat-p2s)
        e_vector = np.sqrt(np.sum(np.square(e_matrix),axis=0))
        inliers = np.argwhere(e_vector<tol).shape[0]

        # save H is it has the largest num. of inliers
        if inliers > max_inliers:
            max_inliers = inliers
            bestH = H2to1
            best_inliers = np.argwhere(e_vector<tol)

    return bestH


def compositeH(H, template, img):
    """
    Returns final warped harry potter image.
    INPUTS
        H - homography
        template - desk image
        img - harry potter image
    OUTPUTS
        final_img - harry potter on book cover image
    """
    m,n,depth = template.shape
    out_size = (n,m)
    img_warped = cv2.warpPerspective(img,H,out_size)

    final_img = copy.deepcopy(template)
    final_img[np.where(img_warped>0)] = 0

    final_img = final_img+img_warped
    cv2.imshow("Final Image", final_img)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()

    return final_img


if __name__ == "__main__":
    im1 = cv2.imread("../data/model_chickenbroth.jpg")
    im2 = cv2.imread("../data/chickenbroth_01.jpg")
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    print(H)
