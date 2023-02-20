import cv2
import numpy as np
from matplotlib import pyplot as plt
from numba import jit, njit
from scipy.ndimage import center_of_mass, label
import time

inicio = time.time()

#@njit
def clip(img):
    xSect = int(np.round_(.28*np.shape(img)[0]))
    ySect = int(np.round_(.15*np.shape(img)[1]))
    newImg = img[xSect:-xSect,ySect:-ySect]
    return newImg, [xSect, ySect]

@njit
def median(img):
    #shp = np.array([np.shape(img)[0]-10, np.shape(img)[1]-10])
    m = np.shape(img)[0]-10
    n = np.shape(img)[1]-10
    new = np.zeros((m,n))
    for r in range(m):
        for c in range(n):
            A = img[r:r+11, c:c+11]
            new[r,c] = np.median(A)
    return new

@jit
def threshold(img):
    suma = 0
    for i in range(256):
        suma += np.count_nonzero(img==i)/(img.shape[0]*img.shape[1])
        if suma > .3:
            T = i
            break
    
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r,c] < T:
                img[r,c] = 0
            else:
                img[r,c] = 255
    """
    img[img<T] = 0
    img[img>=T] = 255
    """
    return img

def IMR(img):
    """
    sh = np.shape(img)
    for r in sh[0]:
        for c in sh[1]:
            if img[r,c] == 0:
    """
    img[img==0] = 1
    img[img==255] = 0
    lbl = label(img)
    #np.seterr(divide='ignore', invalid='ignore')
    cand = np.array(center_of_mass(img, lbl[0], range(lbl[1]))) #candidates to IMR centroid
    #np.seterr(divide='warn', invalid='warn')
    center = np.array([img.shape[0], img.shape[1]])/2.0 #center of image
    dist_to_center = np.sum((cand-np.ones(np.array(cand.shape))*center)**2, axis=1) #distance to center of each candidate
    centroidIdx = np.argmin(dist_to_center[1:])
    img[lbl[0]!=centroidIdx+1] = -1
    img[lbl[0]==centroidIdx+1] = 255
    img[img==-1] = 0
    
    return img

#@jit
def ajuste(arr, perc):
    regionIMR = np.where(arr==255)
    sizeIMR = len(regionIMR[1])
    sampleSize = int(np.floor(perc*sizeIMR))
    #sampleSize = sampleInfo(sizeIMR, perc)
    sampleIdx = np.random.randint(0, sizeIMR, sampleSize)
    sample = np.zeros((2, sampleSize))
    for i in range(sampleSize):
        sample[0, i] = regionIMR[1][sampleIdx[i]]
        sample[1, i] = regionIMR[0][sampleIdx[i]]
    Pn = np.poly1d(np.polyfit(sample[0], sample[1], 2))
    xMin = int(np.min(sample[0]))
    xMax = int(np.max(sample[0]))
    return Pn, [xMin, xMax]
"""
@njit
def sampleInfo(float(size), float(perc)):
    return int(np.floor(perc*size))
"""
@jit
def sobel(img):
    m = np.shape(img)[0]-2
    n = np.shape(img)[1]-2
    new = np.zeros((m,n))
    for r in range(m):
        for c in range(n):
            A = img[r:r+3, c:c+3]
            new[r,c] = kernelSobel(A)
    return new
@njit
def kernelSobel(img):
    Gx = np.array([[1,0,-1],
                   [2,0,-2],
                   [1,0,-1]])
    Gy = np.array([[ 1, 2, 1],
                   [ 0, 0, 0],
                   [-1,-2,-1]])
    sh = np.shape(Gx)
    resX = 0
    resY = 0
    for r in range(sh[0]):
        for c in range(sh[1]):
            resX += img[r,c]*Gx[r,c]
            resY += img[r,c]*Gy[r,c]
    res = np.round_(np.sqrt(resX*resX + resY*resY))
    if res>255:
        res = 255
    return res

#@njit
def curveIntensity(img, poly, offset, imrRange):
    sh = np.shape(img)
    result = []

    for y in range(101):
        totalIntensity = 0
        for c in range(imrRange[0], imrRange[1]+1):
            x = int(np.round_(poly(c)-y-offset[0]))
            if x>0:
                if x<sh[0]:
                    totalIntensity += img[x,c-offset[1]]
        result.append(totalIntensity)
    return result

im1 = '//home//juancho//Documents//Servicio//CIO//139001-17-M.png'
im2 = '//home//juancho//Downloads//138785-21-M.png'
im3 = '//home//juancho//Downloads//138943-20-M.png'
im4 = '//home//juancho//Downloads//139008-22-H.png'

img = cv2.imread(im4, 0)
clippedImg, offset = clip(img)

new = median(img)
new = threshold(new)
new = IMR(new)
poly, imrRange = ajuste(new, .1)

new = sobel(clippedImg)
intensities = curveIntensity(new, poly, offset, imrRange)
intensities1 = curveIntensity(clippedImg, poly, offset, imrRange)
"""
inicio05 = time.time()
poly05, xMin, xMax = ajuste(new, 0.05)
fin05 = time.time()

inicio10 = time.time()
poly10, xMin, xMax = ajuste(new, 0.1)
fin10 = time.time()

inicio20 = time.time()
poly20, xMin, xMax = ajuste(new, 0.2)
fin20 = time.time()

inicio50 = time.time()
poly50, xMin, xMax = ajuste(new, 0.5)
fin50 = time.time()

fin = time.time()
print("t total: ", fin-inicio)
print("t 05: ", fin05-inicio05)
print("t 10: ", fin10-inicio10)
print("t 20: ", fin20-inicio20)
print("t 50: ", fin50-inicio50)
"""

f3 = plt.figure(3)
f3.add_subplot(121)
plt.scatter(range(len(intensities)), intensities)
f3.add_subplot(122)
plt.scatter(range(len(intensities1)), intensities1)

minIntensidadIdx = np.argmin(intensities[np.argmax(intensities):])

f1 = plt.figure(1)
x = np.linspace(imrRange[0], imrRange[1], 1000)
plt.imshow(img, cmap='gray')
plt.plot(x, poly(x))
plt.plot(x, poly(x)-minIntensidadIdx, color="r")

f2 = plt.figure(2)
plt.imshow(new, cmap='gray')
plt.plot(x-offset[0], poly(x)-offset[1])
plt.plot(x-offset[0], poly(x)-offset[1]-minIntensidadIdx, color="r")

plt.show()
#f, a = plt.subplots(1,1)
#x = np.linspace(xMin, xMax, 1000)
#a = plt.plot(x, poly05(x), color="r")
#a = plt.plot(x, poly10(x)+10, color="g")
#a = plt.plot(x, poly20(x)+20, color="b")



'''
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''