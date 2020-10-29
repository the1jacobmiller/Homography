import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite, briefMatch, plotMatches
import matplotlib.pyplot as plt


def imageStitching(im1, im2, H2to1):
    """
    Returns a panorama of im1 and im2 using the given
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    """
    #######################################
    im1_pano = np.zeros((im1.shape[0] + 80, im1.shape[1] + 750, 3), dtype=np.uint8)
    im1_pano[: im1.shape[0], : im1.shape[1], : im1.shape[2]] = im1
    im1_pano_mask = im1_pano > 0

    # warp im2 onto pano
    m,n,depth = im1_pano.shape
    out_size = (n,m)
    pano_im = cv2.warpPerspective(im2,H2to1,out_size)
    pano_im_mask = pano_im > 0

    # dealing with the center where images meet.
    im_center_mask = np.logical_and(im1_pano_mask,pano_im_mask).astype(dtype=np.uint8)

    im_full = pano_im + im1_pano
    im_R = im_full * np.logical_not(im1_pano_mask)
    im_L = im_full * np.logical_not(pano_im_mask)

    # produce im center, mix of pano_im and im1_pano
    im_center = (im1_pano*im_center_mask)/2+(pano_im*im_center_mask)/2
    panorama = im_R + im_L + im_center

    # cv2.imshow("panorama", panorama)
    # cv2.waitKey(0)  # press any key to exit
    # cv2.destroyAllWindows()

    return panorama


def imageStitching_noClip(im1, im2, H2to1):
    """
    Returns a panorama of im1 and im2 using the given
    homography matrix without cliping.
    """
    ######################################
    # TO DO ...
    s = 1
    tx = 0
    # clip
    # establish corners
    # create new corners
    # ty = ... used for M_translate matrix
    m,n,depth = im1.shape
    m2,n2,depth2 = im2.shape
    buffer = 30

    top_left = np.matmul(H2to1, np.array([[0],[0],[1]]))
    top_left = (top_left / top_left[2,:])[0:2,:]
    ty = -1*top_left[1]

    # you actually dont need to use M_scale for the pittsburgh city stitching.
    M_scale = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    M_translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    M = np.matmul(M_scale, M_translate)

    # find best out_size
    H = np.matmul(M,H2to1)
    top_left = H.dot(np.array([[0],[0],[1]]))
    top_left = (top_left / top_left[2,:])[0:2,:]
    top_right = H.dot(np.array([[n2],[0],[1]]))
    top_right = (top_right / top_right[2,:])[0:2,:]
    bottom_right = H.dot(np.array([[n2],[m2],[1]]))
    bottom_right = (bottom_right / bottom_right[2,:])[0:2,:]

    new_n = int(max(bottom_right[0],top_right[0]))
    new_m = int(bottom_right[1])

    out_size = (new_n,new_m)
    pano_im2 = cv2.warpPerspective(im2, np.matmul(M,H2to1), out_size)
    pano_im1 = cv2.warpPerspective(im1, M, out_size)

    im1_pano_mask = pano_im1 > 0
    im2_pano_mask = pano_im2 > 0

    im_center_mask = np.logical_and(im1_pano_mask,im2_pano_mask).astype(dtype=np.uint8)
    pano_im_full = pano_im1 + pano_im2

    im_R = pano_im_full * np.logical_not(im1_pano_mask)
    im_L = pano_im_full * np.logical_not(im2_pano_mask)
    im_center = (pano_im1*im_center_mask)/2+(pano_im2*im_center_mask)/2
    panorama = im_R + im_L + im_center
    return panorama


def generatePanorama(im1, im2):
    H2to1 = np.load("bestH.npy")
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im


if __name__ == "__main__":
    im1 = cv2.imread("../data/incline_L.png")
    im2 = cv2.imread("../data/incline_R.png")

    # im1 = cv2.imread("../data/hi_L.jpg")
    # im2 = cv2.imread("../data/hi_R.jpg")
    print(im1.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc2, desc1)
    # plotMatches(im1,im2,matches,locs2,locs1)
    H2to1 = ransacH(matches, locs2, locs1, num_iter=10000, tol=2)
    # save bestH.npy
    np.save("bestH.npy",H2to1)
    np.save("../results/q7_1.npy",H2to1)

    # pano_im = imageStitching(im1, im2, H2to1)
    # cv2.imwrite("../results/7_1.jpg", pano_im)

    pano_im = generatePanorama(im1, im2)
    cv2.imwrite("../results/7_3.jpg", pano_im)
    cv2.imshow("panoramas", pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
