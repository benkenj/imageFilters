from PIL import Image
import numpy as np
import math
from scipy import signal

def boxfilter(N):
    assert(N % 2 == 1)
    return np.full((N,N),1.0/(float(N)*float(N)))

def gauss1d(sigma):
    fsigma = float(sigma)
    length = int(math.ceil(fsigma * 6) if math.ceil(fsigma * 6) % 2 == 1 else math.ceil(fsigma * 6) + 1)
    gaus = np.arange(-(length-1)/2, ((length-1)/2)+1, 1, dtype=float)
    gausfunc = lambda x: math.exp(-(pow(x,2) / (2.0*pow(fsigma,2))))
    vfunc = np.vectorize(gausfunc)
    return vfunc(gaus)
 
def gauss2d(sigma):
    return signal.convolve2d(gauss1d(sigma)[np.newaxis], np.transpose(gauss1d(sigma)[np.newaxis]))

print(gauss1d(1))
print(gauss2d(1))

    