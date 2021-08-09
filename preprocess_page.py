import cv2
import numpy as np
import math
import copy
from scipy.ndimage import uniform_filter1d
from skimage.filters import threshold_local

np.seterr(all='ignore')

ht, wd = [None]*2

def get_complete_contour(image):
    gray_image = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image],[0],None,[256],[0,256])
    min_idx = np.argmin(np.squeeze(histogram[50:150].T)) + 50
    _, thresholded_image = cv2.threshold(gray_image,min_idx,255,cv2.THRESH_BINARY)
    thresholded_blurry_image = cv2.blur(thresholded_image, (21,21))
    _, thresholded_blurry_image = cv2.threshold(thresholded_blurry_image,min_idx,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresholded_blurry_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cnts_sorted[-1]
    return cnt


def dist_nw(pt):
    global ht, wd
    x1, y1 = pt
    return math.sqrt((0 - x1)**2 + (0 - y1)**2)


def dist_sw(pt):
    global ht, wd
    x1, y1 = pt
    return math.sqrt((0 - x1)**2 + (ht - y1)**2)


def dist_ne(pt):
    global ht, wd
    x1, y1 = pt
    return math.sqrt((wd - x1)**2 + (0 - y1)**2)


def dist_se(pt):
    global ht, wd
    x1, y1 = pt
    return math.sqrt((wd - x1)**2 + (ht - y1)**2)


def get_top_bottom_contour(cnt_list, nw_corner, sw_corner, ne_corner, se_corner):
    new_cnt_list = copy.deepcopy(cnt_list)
    new_cnt_list = new_cnt_list[::-1]
    nw_index = new_cnt_list.index(nw_corner)
    new_cnt_list = new_cnt_list[nw_index:] + new_cnt_list[:nw_index]

    top_cnt_list = new_cnt_list[new_cnt_list.index(nw_corner):new_cnt_list.index(ne_corner)+1]
    
    sw_index = cnt_list.index(sw_corner)
    cnt_list = cnt_list[sw_index:] + cnt_list[:sw_index]

    bottom_cnt_list = cnt_list[cnt_list.index(sw_corner):cnt_list.index(se_corner)+1]
    
    return top_cnt_list, bottom_cnt_list
    

def get_middle(top_cnt_list, bottom_cnt_list):
    top_cnt_np = np.array(top_cnt_list)
    min_x = np.min(top_cnt_np[:, 0])
    max_x = np.max(top_cnt_np[:, 0])
    start = np.abs(top_cnt_np[:, 0] - (min_x + (max_x - min_x)*0.25)).argmin()
    end = np.abs(top_cnt_np[:, 0] - (min_x + (max_x - min_x)*0.75)).argmin()
    top_cnt_middle = top_cnt_np[start:end].copy()
    top_cnt_middle_filt = top_cnt_middle.copy()
    top_cnt_middle_filt[:, 1] = uniform_filter1d(top_cnt_middle_filt[:, 1], size=3, mode='nearest')
    
    top_cnt_grad = np.gradient(top_cnt_middle_filt[:, 1], top_cnt_middle_filt[:, 0])
    top_cnt_grad = np.nan_to_num(top_cnt_grad)
    top_cnt_grad[0] = top_cnt_grad[-1] = 0
    top_cnt_grad = np.around(top_cnt_grad, decimals=8)

    top_middle = None
    peak_grad_idx = np.argmax(np.abs(top_cnt_grad))
    for i in range(peak_grad_idx, top_cnt_middle.shape[0]):
        if top_cnt_grad[i] <= 0:
            top_middle = top_cnt_middle[i].tolist()
            break
    
    bottom_cnt_np = np.array(bottom_cnt_list)
    min_x = np.min(bottom_cnt_np[:, 0])
    max_x = np.max(bottom_cnt_np[:, 0])
    start = np.abs(bottom_cnt_np[:, 0] - (min_x + (max_x - min_x)*0.25)).argmin()
    end = np.abs(bottom_cnt_np[:, 0] - (min_x + (max_x - min_x)*0.75)).argmin()
    bottom_cnt_middle = bottom_cnt_np[start:end].copy()

    bottom_cnt_middle_filt = bottom_cnt_middle.copy()
    bottom_cnt_middle_filt[:, 1] = uniform_filter1d(bottom_cnt_middle_filt[:, 1], size=3, mode='nearest')
    
    bottom_cnt_grad = np.gradient(bottom_cnt_middle_filt[:, 1], bottom_cnt_middle_filt[:, 0])
    bottom_cnt_grad = np.nan_to_num(bottom_cnt_grad)
    bottom_cnt_grad[0] = bottom_cnt_grad[-1] = 0
    bottom_cnt_grad = np.around(bottom_cnt_grad, decimals=8)
    bottom_middle = None
    peak_grad_idx = np.argmax(np.abs(bottom_cnt_grad))
    for i in range(peak_grad_idx, bottom_cnt_middle.shape[0]):
        if bottom_cnt_grad[i] >= 0:
            bottom_middle = bottom_cnt_middle[i].tolist()
            break
    
    return top_middle, bottom_middle


def crop_page(page, top, bottom, left, right):
    return page[top:bottom+1, left:right+1]


def uncurve_page(page, cnt_list, corners):
    nw, ne, sw, se = corners
    
    top_cnt_list = cnt_list[cnt_list.index(nw):cnt_list.index(ne)+1]
    top_cnt = np.expand_dims(np.array(top_cnt_list), 1)
    top_extr = np.min(top_cnt[:, :, 1])
    top_edge = np.zeros(shape=[page.shape[1], 2], dtype=np.int32)
    top_edge[:, 0] = np.array(list(range(0, page.shape[1])))
    top_edge[:, 1] = np.interp(top_edge[:, 0].T, np.squeeze(top_cnt, 1)[:, 0].T, np.squeeze(top_cnt, 1)[:, 1].T, left=top_extr, right=top_extr)
    
    bottom_cnt_list = cnt_list[::-1][cnt_list[::-1].index(sw):cnt_list[::-1].index(se)+1]
    bottom_cnt = np.expand_dims(np.array(bottom_cnt_list), 1)
    bottom_extr = np.max(bottom_cnt[:, :, 1])
    bottom_edge = np.zeros(shape=[page.shape[1], 2], dtype=np.int32)
    bottom_edge[:, 0] = np.array(list(range(0, page.shape[1])))
    bottom_edge[:, 1] = np.interp(bottom_edge[:, 0].T, np.squeeze(bottom_cnt, 1)[:, 0].T, np.squeeze(bottom_cnt, 1)[:, 1].T, left=bottom_extr, right=bottom_extr)
    
    top_offset = top_edge[:, 1].T - top_extr
    bottom_offset = bottom_edge[:, 1].T - bottom_extr

    offset = np.zeros(shape=[top_extr, page.shape[1]], dtype=np.int32)
    xx = np.linspace(top_offset, bottom_offset, bottom_extr-top_extr+1).astype('int32')
    offset = np.concatenate([offset, xx], axis=0)
    offset = np.concatenate([offset, np.zeros(shape=[page.shape[0] - bottom_extr - 1, page.shape[1]], dtype=np.int32)], axis=0)
    
    row_index = np.expand_dims(np.array(list(range(page.shape[0])), dtype=np.int32), axis=0).T
    row_index = np.repeat(row_index, offset.shape[1], axis=1)
    offset_row_index = (row_index + offset) % page.shape[0]
    col_idx = np.array(list(range(page.shape[1])), dtype=np.int32)
    col_idx = np.repeat(np.expand_dims(col_idx, 0), page.shape[0], axis=0)
    
    page_unc = page[offset_row_index, col_idx]
    
    right_extr = max(np.max(top_cnt[:, :, 0]), np.max(bottom_cnt[:, :, 0]))
    left_extr = min(np.min(top_cnt[:, :, 0]), np.min(bottom_cnt[:, :, 0]))

    return crop_page(page_unc, top_extr, bottom_extr, left_extr, right_extr)


def threshold_image(page, threshold):
    if len(page.shape) == 3:
        page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    T = threshold_local(page, threshold, offset = 10, method = "gaussian")
    page = (page > T).astype("uint8") * 255
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def kernel_sharpening(page):
    sharpened = cv2.filter2D(page, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return sharpened


def get_corners(image, cnt_list):
    global ht, wd
    ht, wd, _ = image.shape
    points_nw = sorted(cnt_list, key=dist_nw)
    points_sw = sorted(cnt_list, key=dist_sw)
    points_ne = sorted(cnt_list, key=dist_ne)
    points_se = sorted(cnt_list, key=dist_se)
    nw_corner = points_nw[0]
    sw_corner = points_sw[0]
    ne_corner = points_ne[0]
    se_corner = points_se[0]
    
    return nw_corner, sw_corner, ne_corner, se_corner


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result, rot_mat


def rotate_coordinates(coords_list, rot_mat):
    coords = np.array(coords_list)
    rot_coords = np.dot(rot_mat, np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T).T
    rot_coords = np.around(rot_coords, decimals=0).astype('int32')
    return rot_coords.tolist()


def get_rotation_angle(image):
    cnt = get_complete_contour(image)
    cnt_list = np.squeeze(cnt, 1).tolist()
    nw_corner, sw_corner, ne_corner, se_corner = get_corners(image, cnt_list)
    top_cnt_list, bottom_cnt_list = get_top_bottom_contour(cnt_list.copy(), nw_corner, sw_corner, ne_corner, se_corner)
    top_middle, bottom_middle = get_middle(top_cnt_list, bottom_cnt_list)
    
    angle_radian = math.atan2(-(top_middle[1] - bottom_middle[1]), top_middle[0] - bottom_middle[0])
    angle_degree = 90 - math.degrees(angle_radian)
    return angle_degree


def preprocess(image):
    angle_degree = get_rotation_angle(image)
    image, rot_mat = rotate_image(image, angle_degree)
    
    cnt = get_complete_contour(image)
    cnt_list = np.squeeze(cnt, 1).tolist()
    nw_corner, sw_corner, ne_corner, se_corner = get_corners(image, cnt_list)
    top_cnt_list, bottom_cnt_list = get_top_bottom_contour(cnt_list.copy(), nw_corner, sw_corner, ne_corner, se_corner)
    top_middle, bottom_middle = get_middle(top_cnt_list, bottom_cnt_list)
    
    # contour of left and right page
    cnt_list = cnt_list[::-1]
    cnt_list = cnt_list[cnt_list.index(nw_corner):] + cnt_list[:cnt_list.index(nw_corner)]
    left_cnt_list = cnt_list[cnt_list.index(nw_corner):cnt_list.index(top_middle)+1] \
                            + cnt_list[cnt_list.index(bottom_middle):cnt_list.index(sw_corner)+1]
    right_cnt_list = cnt_list[cnt_list.index(top_middle):cnt_list.index(ne_corner)+1] \
                            + cnt_list[cnt_list.index(se_corner):cnt_list.index(bottom_middle)+1]

    # seperate left and right page
    left_page = image.copy()[:, :top_middle[0] if top_middle[0]>bottom_middle[0] else bottom_middle[0]]
    right_page = image.copy()[:, top_middle[0] if top_middle[0]<bottom_middle[0] else bottom_middle[0]:]
    right_cnt_list = (np.array(right_cnt_list) - np.array([top_middle[0] if top_middle[0]<bottom_middle[0] else bottom_middle[0], 0])).tolist()
    
    nw_left = nw_corner.copy()
    sw_left = sw_corner.copy()
    ne_left = top_middle.copy()
    se_left = bottom_middle.copy()

    nw_right = [top_middle[0] - (top_middle[0] if top_middle[0]<bottom_middle[0] else bottom_middle[0]), top_middle[1]]
    sw_right = [bottom_middle[0] - (top_middle[0] if top_middle[0]<bottom_middle[0] else bottom_middle[0]), bottom_middle[1]]
    ne_right = [ne_corner[0] - (top_middle[0] if top_middle[0]<bottom_middle[0] else bottom_middle[0]), ne_corner[1]]
    se_right = [se_corner[0] - (top_middle[0] if top_middle[0]<bottom_middle[0] else bottom_middle[0]), se_corner[1]]
    
    # uncurve left and right page
    left_page_unc = uncurve_page(left_page, left_cnt_list, [nw_left, ne_left, sw_left, se_left])
    right_page_unc = uncurve_page(right_page, right_cnt_list, [nw_right, ne_right, sw_right, se_right])
    
    # final processing
    left_page_unc_thresh = threshold_image(left_page_unc.copy(), threshold=9)
    right_page_unc_thresh = threshold_image(right_page_unc.copy(), threshold=9)
    left_page_unc_sharp = kernel_sharpening(left_page_unc_thresh)
    right_page_unc_sharp = kernel_sharpening(right_page_unc_thresh)
    
    return left_page_unc, right_page_unc, left_page_unc_sharp, right_page_unc_sharp
