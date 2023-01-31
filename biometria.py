import cv2
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from scipy.ndimage import center_of_mass, label

@jit
def median(img):
    #shp = np.array([np.shape(img)[0]-10, np.shape(img)[1]-10])
    m = np.shape(img)[0]-10
    n = np.shape(img)[1]-10
    new = np.zeros((m,n))
    for r in range(m):
        for c in range(n):
            #A = img[r-5:r+6, c-5:c+6]
            A = img[r:r+11, c:c+11]
            #print(A)
            #new[r,c] = np.median(img[r-5:r+6, c-5:c+6])
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
    cand = np.array(center_of_mass(img, lbl[0], range(lbl[1]))) #candidates to IMR centroid
    center = np.array([img.shape[0], img.shape[1]])/2.0 #center of image
    dist_to_center = np.sum((cand-np.ones(np.array(cand.shape))*center)**2, axis=1) #distance to center of each candidate
    print(dist_to_center)
    centroidIdx = np.argmin(dist_to_center[1:])
    img[lbl[0]!=centroidIdx+1] = -1
    img[lbl[0]==centroidIdx+1] = 0
    img[lbl[0]==-1] = 255
    
    return img
@jit
def norm2(arr):
    a = len(arr)
    norms = np.zeros(a)
    for i in range(a):
        norms[i] = arr[i][0]**2+arr[i][1]**2

img = cv2.imread('C:\\Users\\juanc\\Documents\\Servicio\\CIO\\139001-17-M.png', 0)

new = median(img)
new = threshold(new)
new = IMR(new)
plt.imshow(new, cmap='gray')
plt.show()
'''
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''