import numpy as np
import cv2

from datetime import datetime

def crop_rect(image, rect_coord):
    '''
    rect_coord 1-D array [min_x, min_y, max_x, max_y]
    x for height y for width
    '''
    if len(image.shape) == 3:
        [h, w, depth] = image.shapen
    elif len(image.shape) == 2:
        [h, w] = image.shape
    else:
        print('wrong dim for image in crop_rect()')
    if rect_coord[0] < 0:
        rect_coord[0] = 0
    if rect_coord[1] < 0:
        rect_coord[1] = 0
    if rect_coord[2] > h:
        rect_coord[2] = h
    if rect_coord[3] > w:
        rect_coord[3] = w
    if 'depth' in locals():
        return image[rect_coord[0]:rect_coord[2], rect_coord[1]:rect_coord[3], depth]
    else:
        return image[rect_coord[0]:rect_coord[2], rect_coord[1]:rect_coord[3]]


def visualize_enhance(image):
    avg = np.mean(image)
    image[image < avg] = avg
    image = image - avg
    image = image / np.max(image) * 255
    return image.astype(np.uint8)


def contour_diameter(contours):
    diameter = 0
    if len(contours) == 0:
        print('no contour')
    for contour in contours:
        rect = cv2.minAreaRect(contour.astype(int))
        d = max(rect[1])
        diameter += d
    return diameter


def contour_center(contours):
    pass


def get_plot_by_pixel(x, y, boundaries):
    pass

def angle(v1, v2):
    if v1.any():
        v1_u = v1 / (np.linalg.norm(v1))
    else:
        v1_u = v1
    if v1.any():
        v2_u = v2 / (np.linalg.norm(v2))
    else:
        v2_u = v2
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def ply2xyz(ply_data, pIm, gIm):
    pIm_aligned = pIm[:, 2:]
    ori_shape = gIm.shape
    p_idx = np.where(pIm_aligned.ravel()!=0)
    g_idx = np.where(gIm.ravel()>32)
    total_points_ply = ply_data.elements[0].data['x'].shape[0]
    true_idx = np.intersect1d(p_idx[0], g_idx[0])
    if true_idx.shape[0] != ply_data['vertex'].count:
        raise Exception
    x_im = np.zeros(ori_shape).ravel()
    y_im = np.zeros(ori_shape).ravel()
    z_im = np.zeros(ori_shape).ravel()
    x_im[true_idx] = ply_data['vertex']['x']
    y_im[true_idx] = ply_data['vertex']['y']
    z_im[true_idx] = ply_data['vertex']['z']
    x_im = x_im.reshape(ori_shape)
    y_im = y_im.reshape(ori_shape)
    z_im = z_im.reshape(ori_shape)
    return np.stack([x_im, y_im, z_im], axis=2)

def get_json_info(json_data, sensor='east'):
    json_info = {}
    lemnatec_metadata = json_data['lemnatec_measurement_metadata']
    meta_time = lemnatec_metadata['gantry_system_variable_metadata']['time']
    json_info['date'] = datetime.strptime(meta_time, '%m/%d/%Y %H:%M:%S')
    json_info['scan_distance'] = float(lemnatec_metadata['gantry_system_variable_metadata']['scanDistance [m]'])
    json_info['fov'] = float(lemnatec_metadata['sensor_fixed_metadata']['field of view y [m]'])
    json_info['scan_direction'] = bool(lemnatec_metadata['gantry_system_variable_metadata']['scanIsInPositiveDirection'])
    if lemnatec_metadata['gantry_system_variable_metadata']['scanIsInPositiveDirection'] == 'True':
        json_info['scan_direction'] = True
    else:
        json_info['scan_direction'] = False
    position_x = float(lemnatec_metadata['gantry_system_variable_metadata']['position x [m]'])
    position_y = float(lemnatec_metadata['gantry_system_variable_metadata']['position y [m]'])
    position_z = float(lemnatec_metadata['gantry_system_variable_metadata']['position z [m]'])
    json_info['scanner_position'] = [position_x, position_y, position_z]
    position_x = float(lemnatec_metadata['sensor_fixed_metadata']['scanner '+ sensor + ' location in camera box x [m]'])
    position_y = float(lemnatec_metadata['sensor_fixed_metadata']['scanner '+ sensor + ' location in camera box y [m]'])
    position_z = float(lemnatec_metadata['sensor_fixed_metadata']['scanner '+ sensor + ' location in camera box z [m]'])
    json_info['cambox_position'] = [position_x, position_y, position_z]
    cambox_offset = json_info['cambox_position']
    cambox_offset[1] *= 2
    json_info['scanner_position'] += np.array(cambox_offset)
    if sensor == 'east':
        if json_info['scan_direction']:
            json_info['scanner_position'] += np.array([0.082, 0.4, 0])
        else:
            json_info['scanner_position'] += np.array([0.082, 0.345, 0])
    elif sensor == 'west':
        if json_info['scan_direction']:
            json_info['scanner_position'] += np.array([0.082, -4.23, 0])
        else:
            json_info['scanner_position'] += np.array([0.082, -4.363, 0])
    return json_info

def contour_length(contour):
    c_len = 0
    p_0 = contour[0, :]
    for i in range(1, len(contour)):
        p_1 = contour[i, :]
        seg_len = np.linalg.norm(p_0 - p_1)
        p_0 = p_1
        c_len += seg_len
    return c_len
        
