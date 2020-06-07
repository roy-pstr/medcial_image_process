import cv2
import matplotlib.pyplot as plt
import numpy as np




# functions

def add_gaussian_noise(im, mean=0.0, std=1.0, min=0, max=255):
    """
    :param im: un-noised image
    :param mean: mean of the noise
    :param std: std of the noise
    :return: noised image
    """

    noisy_img = im + np.random.normal(mean, std, im.shape)
    return np.clip(noisy_img, min, max)  # we might get out of bounds due to noise

##################### Threshold functions #####################

def iter_threshold (img):
    h = img.shape[0]
    w = img.shape[1]
    t_init = round(np.mean(img))    #calculating T_init
    t_new = -1
    t_newest = t_init
    while t_newest != t_new:    #repeating the calculation of T_new until it stops changing
        r1_sum = 0
        r1_count = 0
        r2_sum = 0
        r2_count = 0
        t_new = t_newest
        for y in range(0, h):
            for x in range(0, w):
                if img[y, x]<t_newest:  #creating R1
                    r1_sum=r1_sum+img[y, x]
                    r1_count = r1_count+1
                else:   #creating R2
                    r2_sum = r2_sum+img[y, x]
                    r2_count = r2_count+1
        miu1 = r1_sum / r1_count
        miu2 = r2_sum / r2_count
        t_newest = round((miu1 + miu2) / 2) #calculating T_new

    bin_img = binary_image(img,t_newest)

    return (t_newest, bin_img)

def binary_image(img, thershold_val):
    h = img.shape[0]
    w = img.shape[1]
    bin_img = np.zeros((h, w))  # creating the binary image
    for y in range(0, h):
        for x in range(0, w):
            if img[y, x] < thershold_val:
                bin_img[y, x] = 0
            else:
                bin_img[y, x] = 1
    return bin_img


def sobel_edge_detection(img, filter):
   edges = cv2.filter2D(img, -1, filter)
   return iter_threshold(edges)[1]


def four_direction_edge_detector(img):
    h = img.shape[0]
    w = img.shape[1]

    ver_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    ver = sobel_edge_detection(img, ver_filter)

    hor_filter = np.rot90(ver_filter,3)
    hor = sobel_edge_detection(img,hor_filter)


    plus_45_filter = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    plus_45 = sobel_edge_detection(img, plus_45_filter)

    minus_45_filter = np.rot90(plus_45_filter,1)
    minus_45 = sobel_edge_detection(img, minus_45_filter)
    return (hor,ver,plus_45,minus_45)

def LoG(img, sigma):
    return cv2.Laplacian(cv2.GaussianBlur(img, (0, 0), sigma),cv2.CV_64F)

def zero_crossing(img):
    (w,h) = img.shape
    result = np.zeros(img.shape)
    for y in range(0,h-1):
        for x in range(0, w - 1):
            cluster = img[x,y]*img[x:x+2,y:y+2]
            is_negetive_neighboor = cluster<0
            if (is_negetive_neighboor.any()):
                result[x,y]=1
    return result


def offset_indices(seed, size=3):
    seed = seed - np.array([np.int(size/2),np.int(size/2)])
    seeds = np.stack((np.tile(seed[0], (size, size)), np.tile(seed[1], (size, size))))
    row, col = seeds + np.indices((size, size))
    return row, col


def zero_surrounded(img):
    return not (img[0, :].any() or img[-1, :].any() or img[:, 0].any() or img[:, -1].any())


def get4n(point, shape):
    x = point[0]
    y = point[1]
    out = []
    max_x = shape[1]-1
    max_y = shape[0]-1

    #top center
    out_x = x
    out_y = min(max(y-1,0),max_y)
    out.append((out_x,out_y))

    #left
    out_x = min(max(x-1,0),max_x)
    out_y = y
    out.append((out_x,out_y))

    #right
    out_x = min(max(x+1,0),max_x)
    out_y = y
    out.append((out_x,out_y))

    #bottom center
    out_x = x
    out_y = min(max(y+1,0),max_y)
    out.append((out_x,out_y))

    return out

def region_growing(img, seed, threshold=0):
    w = img.shape[0]
    h = img.shape[1]
    seg = np.zeros((w, h))
    seed_val = img[seed]
    threshold_map = np.abs(img.astype(int) - seed_val)
    bin_img = np.where(threshold_map <= threshold, 1, 0)
    size_in_pixels = 0
    already_checked = []
    points_lst = []
    points_lst.append((seed[0], seed[1]))
    while len(points_lst) > 0:
        curr_pix = points_lst[0]
        seg[curr_pix] = 255
        for coord in get4n(curr_pix, img.shape):
            if bin_img[coord] == 1:
                seg[coord] = 255
                if coord not in already_checked:
                    points_lst.append(coord)
                already_checked.append(coord)
        points_lst.pop(0)

    size_in_pixels = np.count_nonzero(seg)
    return (seg.astype(int), size_in_pixels)