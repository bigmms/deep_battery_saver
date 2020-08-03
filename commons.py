import cv2
import math
from scipy import signal
import scipy.ndimage
import numpy as np


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def bg_adjust( bg_lum0, min_lum ):
    row,col = np.shape(bg_lum0)
    bg_lum = bg_lum0
    for x in range(row):
        for y in range(col):
            if bg_lum[x,y]<=127:
                bg_lum[x,y] = np.around(min_lum + bg_lum(x,y)*(127-min_lum)/127)
    return bg_lum


def luminance_contrast(img):
    R = 2
    ker = np.ones((2 * R + 1,2 * R + 1)) / (2 * R + 1)**2
    kern = ker.tolist()
    mean_mask = scipy.ndimage.correlate(img, ker, mode='nearest')
    mean_img_sqr = mean_mask**2
    img_sqr = img**2
    mean_sqr_img = scipy.ndimage.correlate(img_sqr, ker, mode='nearest')
    var_mask = mean_sqr_img - mean_img_sqr
    var_mask[var_mask < 0] = 0
    row, col = np.shape(img)
    valid_mask = np.zeros((row, col))
    valid_mask[R: - R, R: - R] = 1
    var_mask = np.multiply(var_mask, valid_mask)
    L_c = np.sqrt(var_mask)
    return L_c


def luminance_contrast(img):
    R = 2
    ker = np.ones((2 * R + 1,2 * R + 1)) / (2 * R + 1)**2
    kern = ker.tolist()
    mean_mask = scipy.ndimage.correlate(img, ker, mode='nearest')
    mean_img_sqr = mean_mask**2
    img_sqr = img**2
    mean_sqr_img = scipy.ndimage.correlate(img_sqr, ker, mode='nearest')
    var_mask = mean_sqr_img - mean_img_sqr
    var_mask[var_mask < 0] = 0
    row, col = np.shape(img)
    valid_mask = np.zeros((row, col))
    valid_mask[R: - R, R: - R] = 1
    var_mask = np.multiply(var_mask, valid_mask)
    L_c = np.sqrt(var_mask)
    return L_c


def bg_lum_jnd(img0):
    bg = func_bg(img0)
    T0 = 17
    gamma = 3 / 128
    col, row = np.shape(img0)
    JNDl = np.zeros((col, row))
    for i in range(col):
        for j in range(row):
            JNDl[i,j] = gamma*(bg[i,j]-127) + 3
            if bg[i,j]<=127:
                JNDl[i, j] = T0*(1-(bg[i,j]/127)**0.5)+3
    JNDl = gamma*(bg-127) + 3


    min_lum = 32
    alpha = 0.7
    B = [[1, 1, 1, 1, 1],
         [1, 2, 2, 2, 1],
         [1, 2, 0, 2, 1],
         [1, 2, 2, 2, 1],
         [1, 1, 1, 1, 1]]
    bg_lum0 = signal.correlate2d(img0, B, mode='same')/32
    bg_lum0 = np.pad(bg_lum0, 4,'symmetric')
    bg_lum = bg_adjust(bg_lum0, min_lum)
    col, row = np.shape(img0)
    bg_jnd = np.zeros([256, 1])
    T0 = 17
    gamma = 3 / 128
    for i in range(1,256):
        lum = i-1
        if lum<=127:
            bg_jnd[i] = T0 * (1 - math.sqrt(lum / 127)) + 3
        else:
            bg_jnd[i] = gamma * (lum - 127) + 3
    jnd_lum = np.zeros([col,row])
    for x in range(col):
        for y in range(row):
            jnd_lum[x, y] = bg_jnd[int(bg_lum[x, y]) + 1]
    jnd_lum_adapt = alpha * jnd_lum
    return jnd_lum_adapt


def func_bg_adjust(bg_lum0, min_lum):
    col, row = np.shape(bg_lum0)
    bg_lum = bg_lum0.copy()
    for x in range(col):
        for y in range(row):
            if bg_lum[x, y] <= 127:
                bg_lum[x, y] = np.round(min_lum + bg_lum[x, y]*(127-min_lum)/127)
    return bg_lum


def func_bg(img0):
    min_lum = 32
    alpha = 0.7
    B = [[1, 1, 1, 1, 1],
         [1, 2, 2, 2, 1],
         [1, 2, 0, 2, 1],
         [1, 2, 2, 2, 1],
         [1, 1, 1, 1, 1]]
    bg_lum0 = np.floor(signal.correlate2d(img0, B, mode='same') / 32)
    bg_lum = func_bg_adjust(bg_lum0, min_lum)
    col, row = np.shape(img0)
    bg_jnd = np.zeros((256, 1))
    T0 = 17
    gamma = 3 / 128
    for i in range(1, 257):
        lum = i - 1
        if lum <= 127:
            bg_jnd[lum,0] = T0 * (1 - math.sqrt(lum / 127)) + 3
        else:
            bg_jnd[lum,0] = gamma * (lum - 127) + 3
    jnd_lum = np.zeros((col, row))
    for x in range(col):
        for y in range(row):
            jnd_lum[x,y] = bg_jnd[np.int(bg_lum[x,y]),0]
    jnd_lum_adapt = alpha * jnd_lum
    return jnd_lum_adapt


def func_luminance_contrast(img):
    R = 2
    ker = np.ones((2 * R + 1, 2 * R + 1)) / (2 * R + 1) ** 2
    mean_mask = scipy.ndimage.correlate(img, ker, mode='constant')
    mean_img_sqr = mean_mask ** 2
    img_sqr = img ** 2
    mean_sqr_img = scipy.ndimage.correlate(img_sqr, ker, mode='constant')
    var_mask = mean_sqr_img - mean_img_sqr
    var_mask[var_mask < 0] = 0
    row, col = np.shape(img)
    valid_mask = np.zeros((row, col))
    valid_mask[R: - R, R: - R] = 1
    var_mask = np.multiply(var_mask, valid_mask)
    L_c = np.sqrt(var_mask)
    return L_c


def func_cmlx_num_compute(img):
    r = 1
    nb = r * 8
    otr = 6
    kx = np.array([[-1, 0, 1],[-1, 0, 1], [-1, 0, 1]])/3.0
    ky = kx.transpose()

    sps = np.zeros((nb,2))
    asa = 2 * math.pi / nb
    for i in range(nb):
        sps[i,0] =  -r * math.sin((i)*asa)
        sps[i, 1] = r * math.cos((i) *asa)
    imgd = np.lib.pad(img, r, 'symmetric')
    row,col = imgd.shape
    Gx = scipy.ndimage.correlate(imgd, kx, mode='constant')
    Gy = scipy.ndimage.correlate(imgd, ky, mode='constant')
    Cimg = (Gx**2 + Gy**2)**0.5
    Cvimg = np.zeros((row, col))
    Cvimg[Cimg >= 5] = 1
    Oimg = np.zeros((row, col))
    for x in range(row):
        for y in range(col):
            Oimg[x,y] = np.round(math.atan2(Gy[x,y], Gx[x,y]) / math.pi * 180)
    Oimg[Oimg > 90] = Oimg[Oimg > 90] - 180
    Oimg[Oimg < -90] = 180 + Oimg[Oimg < -90]
    Oimg = Oimg + 90
    Oimg[Cvimg == 0] = 180 + 2 * otr
    Oimgc = Oimg[r :row - r, r : col - r ]
    Cvimgc = Cvimg[r :row - r, r : col - r ]

    Oimg_norm = np.round(Oimg / 2 / otr)
    Oimgc_norm = np.round(Oimgc / 2 / otr)
    onum = np.round(180 / 2 / otr) + 1

    ssr_val = np.zeros((row - 2 * r, col - 2 * r, np.uint8(onum) + 1))
    for x in range(int(onum)+1):
        Oimgc_valid = Oimgc_norm == x
        ssr_val[:,:, x] = ssr_val[:,:, x] + Oimgc_valid
    for i in range(nb):
        dx = int(np.round(r + sps[i, 0]))
        dy = int(np.round(r + sps[i, 1]))
        Oimgn = Oimg_norm[dx:row - 2 * r + int(dx), dy: col - 2 * r + int(dy)]
        for x in range(int(onum)+1):
            Oimg_valid = Oimgn == x
            ssr_val[:,:, x] = ssr_val[:,:, x] + Oimg_valid

    ssr_no_zero = ssr_val != 0
    cmlx = np.sum(ssr_no_zero, axis = 2)
    cmlx[Cvimgc == 0] = 1
    cmlx[0: r,: ] = 1
    cmlx[- r : -1,: ] = 1
    cmlx[:, 0: r ] = 1
    cmlx[:, - r: -1] = 1
    return cmlx


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def func_ori_cmlx_compute(img):
    cmlx_map = func_cmlx_num_compute(img)
    r = 3
    sig = 1
    fker = fspecial_gauss(r, sig)
    cmlx_mat = scipy.ndimage.correlate(cmlx_map.astype(float), fker, mode='constant')

    return cmlx_mat


def func_edge_height(img):
    G1 = [[0, 0, 0, 0, 0],
          [1, 3,8,3,1],
          [0, 0, 0, 0, 0],
          [- 1, - 3, - 8, - 3, - 1],
          [0, 0, 0, 0, 0]]
    G2 = [[0, 0, 1, 0, 0],
          [0,8, 3, 0, 0],
          [1, 3, 0, - 3, - 1],
          [0, 0, - 3, - 8, 0],
          [0, 0, - 1, 0, 0]]
    G3 = [[0, 0, 1, 0, 0],
          [0, 0, 3, 8, 0],
          [- 1, - 3, 0, 3,1],
          [0, - 8, - 3, 0, 0],
          [0, 0, - 1, 0, 0]]
    G4 = [[0, 1, 0, - 1, 0],
          [0, 3, 0, - 3, 0],
          [0, 8, 0, - 8, 0],
          [0, 3, 0, - 3, 0],
          [0, 1, 0, - 1, 0]]
    size_x, size_y = img.shape
    grad = np.zeros((size_x, size_y, 4))
    grad[:,:, 0] = signal.correlate2d(img, G1, mode='same')/ 16
    grad[:, :, 1] = signal.correlate2d(img, G2, mode='same') / 16
    grad[:, :, 2] = signal.correlate2d(img, G3, mode='same') / 16
    grad[:, :, 3] = signal.correlate2d(img, G4, mode='same') / 16
    max_gard = np.max(np.abs(grad), axis = 2)
    maxgard = max_gard[2:- 2, 2: - 2]
    edge_height = np.lib.pad(maxgard, 2, 'symmetric')
    return edge_height


def func_edge_protect(img):
    edge_h = 60
    edge_height = func_edge_height(img)
    max_val = np.max(edge_height[:])
    edge_threshold = edge_h / max_val
    if edge_threshold > 0.8:
        edge_threshold = 0.8
    # edge_region = edge(img, 'canny', edge_threshold)
    smoothedInput = cv2.GaussianBlur(img, (3, 3), 0)
    xgrad = cv2.Sobel(smoothedInput,cv2.CV_16SC1,1,0)
    ygrad = cv2.Sobel(smoothedInput,cv2.CV_16SC1,0,1)
    edge_region = cv2.Canny(xgrad,ygrad, 50, 150)
    se = np.ones((5,5))
    img_edge = cv2.dilate(edge_region, se)
    img_edge[img_edge == 255]=1
    img_supedge = 1 - 1 * (img_edge)
    gaussian_kernal = fspecial_gauss(5, 0.8)
    edge_protect = signal.correlate2d(img_supedge, gaussian_kernal, mode='same')
    return edge_protect


def get_jnd_map(img):
    jnd_LA = func_bg(img)
    L_c = func_luminance_contrast(img)
    a1 = 0.115*16
    a2 = 26
    jnd_LC = (a1 * L_c**2.4)/ (L_c**2 + a2**2)
    P_c = func_ori_cmlx_compute(img)
    a3 = 0.3
    a4 = 2.7
    a5 = 1
    C_t = (a3 * P_c**a4)/ (P_c**2 + a5**2)
    jnd_PM = L_c* C_t
    edge_protect = func_edge_protect(img)
    jnd_PM_p = jnd_PM * edge_protect
    d = np.array((jnd_LC, jnd_PM_p))
    jnd_VM = d.max(axis=0)
    jnd_map = jnd_LA + jnd_LC - 0.3 * np.minimum(jnd_LA, jnd_VM)
    return jnd_map