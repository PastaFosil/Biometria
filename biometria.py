import cv2
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from scipy.ndimage import center_of_mass, label
import time

inicio = time.time()
@jit
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

def threshold(img):
    suma = 0
    for i in range(256):
        suma += np.count_nonzero(img==i)/(img.shape[0]*img.shape[1])
        if suma > .3:
            T = i
            break
    img[img<T] = 0
    img[img>=T] = 255
    
    return img

def IMR(img):
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
@jit
def norm2(arr):
    a = len(arr)
    norms = np.zeros(a)
    for i in range(a):
        norms[i] = arr[i][0]**2+arr[i][1]**2

def ajuste(arr, perc):
    regionIMR = np.where(arr==255)
    sizeIMR = len(regionIMR[1])
    sampleSize = int(np.floor(perc*sizeIMR))
    print("sampleSize = ", sampleSize)
    sampleIdx = np.random.randint(0, sizeIMR, sampleSize)
    sample = np.zeros((2, sampleSize))
    for i in range(sampleSize):
        sample[0, i] = regionIMR[1][sampleIdx[i]]
        sample[1, i] = regionIMR[0][sampleIdx[i]]
    Pn = np.poly1d(np.polyfit(sample[0], sample[1], 2))
    xMin = np.min(sample[0])
    xMax = np.max(sample[0])
    return Pn, xMin, xMax

img = cv2.imread('//home//juancho//Documents//Servicio//CIO//139001-17-M.png', 0)

new = median(img)
new = threshold(new)
new = IMR(new)

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


f, a = plt.subplots(1,1)
x = np.linspace(xMin, xMax, 1000)
#a = plt.plot(x, poly05(x), color="r")
#a = plt.plot(x, poly10(x)+10, color="g")
a = plt.plot(x, poly20(x)+20, color="b")
a = plt.imshow(img, cmap='gray')
plt.show()


'''
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''