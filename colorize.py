import warnings
from os.path import normpath as fn  # Fixes window/linux path conventions
from scipy.signal import convolve2d as conv2
from scipy import sparse
from scipy.sparse import linalg
from scipy.interpolate import interp2d
import numpy as np
import cv2
import math
from skimage.io import imread, imsave
################ COLOR RESTORATION FUNCTIONS:


def getColor(colorIm, yuvIm, isGif=False):
    n,m,l = yuvIm.shape
    size = n * m

    inds = np.arange(size).reshape(n, m, order="F") # Indices matrix counting by column
    outputIm = np.zeros_like(yuvIm)
    outputIm[:,:,0] = yuvIm[:,:,0]

    w = 1 # Neighborhood window radius
    windCount = (2 * w + 1) ** 2 # number of pixels in window
    maxSamples = size * windCount
    rowInd = np.zeros(maxSamples).astype(np.int64)
    colInd = np.zeros(maxSamples).astype(np.int64)
    vals = np.zeros(maxSamples)

    length = 0
    pxl = 0

    for j in range(m):
        for i in range(n):

            # if the current pixel does not have a color marked, iterate over pixels in window around it
            if not colorIm[i,j]:
                wInd = 0 # window index
                wVals = np.zeros(maxSamples) #window values

                for k in range(max(0,i-w), min(i+w+1, n)):
                    for l in range(max(0,j-w), min(j+w+1, m)):

                        # If current pixel in window is not center pixel of window
                        if (k != i) or (l != j):
                            # set row and column ind values to be pixel number and index
                            rowInd[length] = pxl
                            colInd[length] = inds[k,l]
                            # set window vals element at window index to intensity of grayscale image at that pixel
                            wVals[wInd] = yuvIm[k,l,0]
                            length += 1
                            wInd += 1

                center = yuvIm[i,j,0].copy() # center is intensity of grayscale at i,j
                wVals[wInd] = center

                wValSlice = wVals[0:wInd+1]
                var = np.mean((wValSlice - np.mean(wValSlice))**2) # variance of intensities in the window
                sigma = var * .6

                sqDiff = (wValSlice-center) ** 2

                mgv = min(sqDiff)
                if sigma < (-mgv/np.log(.01)): sigma = -mgv / np.log(.01)
                if sigma < .000002: sigma = .000002

                #print(wInd)
                weight = np.exp(-((wVals[0:wInd] - center) ** 2)/sigma)
                wVals[0:wInd] = weight
                # normalize weights
                wVals[0:wInd] /= np.sum(wVals[0:wInd])
                vals[length-wInd:length] = -wVals[0:wInd]

            # now pixels are colored, set row and column inds and increment pxl and length
            rowInd[length] = pxl
            colInd[length] = inds[i,j]
            vals[length] = 1
            pxl += 1
            length += 1

    vals = vals[0:length]
    colInd = colInd[0:length]
    rowInd = rowInd[0:length]

    # csr_matrix((data, (row_ind, col_ind)), [shape=(M,N)])
    A = sparse.csr_matrix((vals, (rowInd,colInd)), (pxl, size))
    b = np.zeros(A.shape[0])
    # get colored indices for solver
    colored = np.nonzero(colorIm.reshape(size,order="F"))

    # Solve sparse linear equation
    for i in [1,2]:
        current = yuvIm[:,:,i].reshape(size,order="F")
        b[colored] = current[colored]
        x = linalg.spsolve(A,b)
        outputIm[:,:,i] = x.reshape(n,m,order="F")

    # outputIm = YIQ2RGB(outputIm)
    return outputIm



fx = np.float32([[1,0,-1]]) * np.float32([[1,1,1]]).T / 6.
fy = fx.T
def lucaskanade(f1,f2,W):
    K = np.ones((W,W)) # kernel of ones in order to use convolution for quick sum
    avg = (f1 + f2) / 2.0
    T = f2 - f1
    stab = .00001

    X = conv2(avg, fx, mode="same")
    Y = conv2(avg, fy, mode="same")

    A11 = conv2(np.square(X), K, mode="same") + stab
    A22 = conv2(np.square(Y), K, mode="same") + stab
    A12 = conv2(X*Y, K, mode="same")
    A21 = conv2(Y*X, K, mode="same")

    # compute inverse of 2x2 matrix pointwise
    det = (A11 * A22) - (A12 * A21)
    A11_i = A22 / det
    A22_i = A11 / det
    A12_i = (-1 * A12) / det
    A21_i = (-1 * A21) / det


    B1 = -1 * conv2(X*T, K, mode="same")
    B2 = -1 * conv2(Y*T, K, mode="same")

    u = A11_i * B1 + A12_i * B2
    v = A21_i * B1 + A22_i * B2


    return np.dstack([u,v])

def RGB2YIQ(RGB):
    ret = np.zeros_like(RGB)
    r = RGB[:,:,0]
    g = RGB[:,:,1]
    b = RGB[:,:,2]
    y = .299 * r + .587 * g + .114 * b
    i = .5959 * r + (-.2746 * g) + (-.3213 * b)
    q = .2115 * r + (-.5227 * g) + .3112 * b
    y[y < 0] = 0
    y[y > 1] = 1
    ret[:,:,0] = y
    i[i < -.5957] = -.5957
    i[i > .5957] = .5957
    ret[:, :, 1] = i
    q[q < -.5226] = -.5226
    q[q > .5226] = .5226
    ret[:, :, 2] = q

    return ret

def YIQ2RGB(YIQ):
    ret = np.zeros_like(YIQ)
    ret[:,:,0] = YIQ[:,:,0] + .956 * YIQ[:,:,1] + .619 * YIQ[:,:,2]
    ret[:,:,1] = YIQ[:,:,0] + (-.272 * YIQ[:,:,1]) + (-.647 * YIQ[:,:,2])
    ret[:,:,2] = YIQ[:,:,0] + (-1.106 * YIQ[:,:,1]) + 1.703 * YIQ[:,:,2]
    ret[ret<0] = 0
    ret[ret>1] = 1
    return ret
################# SETUP CODE:


################## Still image solver:
def still(grayscaleName='example.bmp', markedName='example_marked.bmp'):
    # grayscaleName = 'example.bmp'
    # markedName = 'example_marked.bmp'
    cleanName = grayscaleName[:grayscaleName.index('.')]
    outputName = "{}_output.jpg".format(cleanName)

    grayIm = np.float32(imread(fn(grayscaleName)))/255.
    markedIm = np.float32(imread(fn(markedName)))/255.
    if len(grayIm.shape) == 2: grayIm = np.dstack([grayIm for x in range(3)])
    isColored = (np.sum(np.abs(grayIm-markedIm), axis=2) > .01)

    # grayIm_conv = cv2.cvtColor(grayIm, cv2.COLOR_RGB2HSV)
    # markedIm_conv = cv2.cvtColor(markedIm, cv2.COLOR_RGB2YUV)
    grayIm_conv = RGB2YIQ(grayIm)
    markedIm_conv = RGB2YIQ(markedIm)

    yuvIm = np.dstack([grayIm_conv[:,:,0], markedIm_conv[:,:,1], markedIm_conv[:,:,2]])
    yuvH, yuvW, yuvD = yuvIm.shape

    # dMax = math.floor(math.log(min(yuvH, yuvW)) / math.log(2) - 2)
    # xa = 0
    # ya = 0
    # xb = math.floor(yuvH / (2 ** (dMax-1))) * (2 ** (dMax -1))
    # yb = math.floor(yuvW / (2 ** (dMax-1))) * (2 ** (dMax -1))

    # isColored = isColored[xa:xb, ya:yb]
    # yuvIm = yuvIm[xa:xb, ya:yb, :]

    outputImYIQ = getColor(isColored, yuvIm)
    outputIm = YIQ2RGB(outputImYIQ)
    imsave(fn(outputName), outputIm)

def gif(grayFrame1='example.bmp', grayFrame2='exampleShift.bmp', markedFrame='example_marked.bmp', THRESH=5):
    grayIm1 = np.float32(imread(fn(grayFrame1)))/255.
    grayIm2 = np.float32(imread(fn(grayFrame2)))/255.
    markedIm = np.float32(imread(fn(markedFrame)))/255.
    isColored1 = (np.sum(np.abs(grayIm1-markedIm), axis=2) > .01)
    flow = lucaskanade(grayIm1[:, :, 0], grayIm2[:, :, 0], 11)
    R2 = np.dstack(np.meshgrid(np.arange(flow.shape[1]),np.arange(flow.shape[0])))
    pxlMap = R2 + flow
    newMarks = cv2.remap(markedIm, pxlMap.astype(np.float32), None, interpolation=cv2.INTER_CUBIC)
    outputmapname = "{}_MAPPING.bmp".format(grayFrame1[:grayFrame1.index('.')])
    imsave(fn(outputmapname), newMarks)
    flowNorm = np.zeros_like(grayIm1[:,:,0])
    flowNorm = np.sqrt(flow[:,:,0] ** 2 + flow[:,:,1] ** 2)
    flowNorm = np.dstack([flowNorm for x in range(3)])
    #neighbors = np.linalg.norm(grayIm1-flowNorm+grayIm2,axis=2) <= THRESH
    neighborsEqn = np.sqrt((grayIm1[:,:,0]+flow[:,:,0]-grayIm2[:,:,0])**2 + (grayIm1[:,:,0]+flow[:,:,1]-grayIm2[:,:,0])**2)
    neighbors = np.where(neighborsEqn <= THRESH, 1, 0)
    isColored2 = (np.sum(np.abs(grayIm2-markedIm), axis=2) > .01) * neighbors

    grayIm1_conv = RGB2YIQ(grayIm1)
    grayIm2_conv = RGB2YIQ(grayIm2)
    markedIm_conv = RGB2YIQ(markedIm)

    yuvIm1 = np.dstack([grayIm1_conv[:, :, 0], markedIm_conv[:, :, 1], markedIm_conv[:, :, 2]])
    yuvIm2 = np.dstack([grayIm2_conv[:, :, 0], markedIm_conv[:, :, 1], markedIm_conv[:, :, 2]])

    outputImYIQ1 = getColor(isColored1, yuvIm1)
    marks2 = getColor(isColored2, yuvIm2)
    marks2RGB = YIQ2RGB(marks2)
    outputIm2YIQ = getColor((np.sum(np.abs(grayIm2-marks2RGB), axis=2) > .01), yuvIm2)


    outputName1 = "{}_out.bmp".format(grayFrame1[:grayFrame1.index('.')])
    outputName2 = "{}_out.bmp".format(grayFrame2[:grayFrame2.index('.')])
    outputIm1 = YIQ2RGB(outputImYIQ1)
    outputIm2 = YIQ2RGB(outputIm2YIQ)
    imsave(fn(outputName1), outputIm1)
    imsave(fn(outputName2), outputIm2)

def main():
    #gif()
    #gif('frame10.jpg','frame11.jpg', 'frame10_marked.jpg')
    #still('trident_gray.bmp', 'trident_marked.bmp')
    name1 = 'kid_marked.bmp'
    name1_1 = 'kid_marked.jpg'
    name2 = 'trident_marked.bmp'
    name2_1 = 'trident_marked.jpg'
    colorIm1 =np.float32(imread(fn(name1))) / 255.
    # colorIm2 = np.float32(imread(fn(name2))) / 255.
    # grayIm = cv2.cvtColor(colorIm, cv2.COLOR_RGB2GRAY)
    imsave(fn(name1_1),colorIm1)
    # imsave(fn(name2_1),colorIm2)
    # print(colorIm.shape)

if __name__ == '__main__':
    main()