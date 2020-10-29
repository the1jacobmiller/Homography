import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4]):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max() > 10:
        im = np.float32(im) / 255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0 * k ** i
        im_pyramid.append(cv2.GaussianBlur(im, (0, 0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(
        im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    cv2.imshow("Pyramid of image", im_pyramid)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1, 0, 1, 2, 3, 4]):
    """
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    """
    DoG_pyramid = []
    for i in range(1,len(levels)):
        DoG = gaussian_pyramid[:,:,i] - gaussian_pyramid[:,:,i-1]
        DoG_pyramid.append(DoG)

    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    """
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid

    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each
                          point contains the curvature ratio R for the
                          corresponding point in the DoG pyramid
    """
    principal_curvature = None

    gxx = []
    gyy = []
    gxy = []
    gyx = []

    for l in range(DoG_pyramid.shape[2]):
        # Computing 1st order derivatives
        gx = cv2.Sobel(
            DoG_pyramid[:, :, l],
            cv2.CV_64F,
            1,
            0,
            ksize=3,
            borderType=cv2.BORDER_CONSTANT,
        )
        gy = cv2.Sobel(
            DoG_pyramid[:, :, l],
            cv2.CV_64F,
            0,
            1,
            ksize=3,
            borderType=cv2.BORDER_CONSTANT,
        )

        # Computing 2nd order derivatives
        gxx.append(
            cv2.Sobel(gx, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )
        gxy.append(
            cv2.Sobel(gx, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )
        gyx.append(
            cv2.Sobel(gy, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )
        gyy.append(
            cv2.Sobel(gy, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        )

    gxx = np.stack(gxx, axis=-1)
    gxy = np.stack(gxy, axis=-1)
    gyx = np.stack(gyx, axis=-1)
    gyy = np.stack(gyy, axis=-1)

    principal_curvature = np.divide(
        np.square(np.add(gxx, gyy)), (np.multiply(gxx, gyy) - np.multiply(gxy, gyx))
    )

    return principal_curvature


def getLocalExtrema(
    DoG_pyramid, DoG_levels, principal_curvature, th_contrast=0.03, th_r=12
):
    """
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    """

    DoG_pyramid = (np.abs(DoG_pyramid)>th_contrast)*np.abs(DoG_pyramid)
    principal_curvature = (np.abs(principal_curvature)<th_r)*principal_curvature

    m,n,l = DoG_pyramid.shape
    locs = []
    for i in range(1,m-1,3):
        for j in range(1,n-1,3):
            for k in range(1,l-1,3):
                window = DoG_pyramid[i-1:i+2,j-1:j+2,k-1:k+2]
                local_max = np.unravel_index(np.argmax(window),window.shape)
                local_max = np.array(local_max) + np.array([i-1,j-1,k-1])
                pixel_value = DoG_pyramid[local_max[0],local_max[1],local_max[2]]
                r = principal_curvature[local_max[0],local_max[1],local_max[2]]
                if pixel_value!=0 and r!=0:
                    x = local_max[1]
                    y = local_max[0]
                    level = DoG_levels[local_max[2]]
                    locs.append([x,y,level])
    locsDoG = np.array(locs)
    return locsDoG


def DoGdetector(
    im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4], th_contrast=0.03, th_r=12
):
    """
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    """

    gaussian_pyramid = createGaussianPyramid(im, sigma0=sigma0, k=k, levels=levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gaussian_pyramid, levels=levels)
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast=th_contrast, th_r=th_r)

    return locsDoG, gaussian_pyramid


if __name__ == "__main__":
    # test gaussian pyramid
    levels = [-1, 0, 1, 2, 3, 4]
    im = cv2.imread("../data/hp_cover.jpg")
    # im = cv2.imread("../data/model_chickenbroth.jpg")
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)

    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    for i in range(locsDoG.shape[0]):
        im = cv2.circle(im, (locsDoG[i,0],locsDoG[i,1]), 1, (0,255,0), 2)
    cv2.imshow("Detected Features", im)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()
    cv2.imwrite('../results/1_5.jpg',im)
