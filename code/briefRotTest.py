import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import BRIEF

im1 = cv2.imread('../data/model_chickenbroth.jpg')
im2 = cv2.imread('../data/chickenbroth_01.jpg')
if len(im1.shape) == 3:
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
if len(im2.shape) == 3:
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
rows,cols = im2.shape

num_matches = []
for theta in range(0,360,10):
    R = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    rotated_im2 = cv2.warpAffine(im2,R,(cols,rows))

    locs1, desc1 = BRIEF.briefLite(im1)
    locs2, desc2 = BRIEF.briefLite(rotated_im2)
    matches = BRIEF.briefMatch(desc1, desc2)
    num_matches.append(len(matches))
    
num_matches = np.array(num_matches)

plt.title("Rotated BRIEF Descriptor Matches")
plt.bar(list(range(0,360,10)), num_matches)
plt.show()
