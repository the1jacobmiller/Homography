import numpy as np
import cv2
import skimage.io
from BRIEF import briefLite, briefMatch, plotMatches
from planarH import computeH, ransacH, compositeH

# warp harry potter onto cv desk image

# read images
im1 = cv2.imread("../data/hp_cover.jpg")
im2 = cv2.imread("../data/pf_scan_scaled.jpg")
im3 = cv2.imread("../data/pf_desk.jpg")

# calculate brief features
locs1, desc1 = briefLite(im1)
locs2, desc2 = briefLite(im2)
locs3, desc3 = briefLite(im3)

# calculate matches
matches12 = briefMatch(desc1, desc2)
matches23 = briefMatch(desc2, desc3)
matches13 = briefMatch(desc1, desc3)

# calculate homography - transform pf_scan_scaled to pf_desk
H3to2 = ransacH(matches23, locs2, locs3, num_iter=5000, tol=2)

# calculate homography - transform hp_cover to pf_scan_scaled
p1 = np.array([[0,im1.shape[1]-1,0,im1.shape[1]-1],[0,0,im1.shape[0]-1,im1.shape[0]-1]])
p2 = np.array([[0,im2.shape[1]-1,0,im2.shape[1]-1],[0,0,im2.shape[0]-1,im2.shape[0]-1]])
H2to1 = computeH(p1, p2)

# compute final warp
H = H3to2.dot(H2to1)
final_img = compositeH(H, template=im3, img=im1)

# save final image as final_image
print(H)
cv2.imwrite('../results/6_1.jpg',final_img)

num_iters = [10,100,500,5000]
tolerances = [0.1, 1, 3, 5, 10]
for num_iter in num_iters:
    for tol in tolerances:
        H3to2 = ransacH(matches23, locs2, locs3, num_iter=num_iter, tol=tol)

        # compute final warp
        H = H3to2.dot(H2to1)
        final_img = compositeH(H, template=im3, img=im1)

        # save final image as final_image
        file_path = '../results/6_2/'+str(num_iter)+'_'+str(tol)+'.jpg'
        cv2.imwrite(file_path,final_img)
