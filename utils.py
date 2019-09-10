import numpy as np
import cv2
import math
from skimage.measure import regionprops
from skimage import filters
from datetime import datetime
from scipy import stats

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
        pass
        # print('wrong dim for image in crop_rect()')
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
    true_idx = np.where((pIm_aligned.ravel() != 0) & (gIm.ravel() > 32))[0]
    if true_idx.shape[0] != ply_data['vertex'].count:
        raise Exception('Number of point from ply data does not match with calculated by raw data!')
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

def corrupt_pixel_ratio(pIm, gIm):
    pIm_aligned = pIm[:, 2:]
    total_pixel_count = pIm_aligned.shape[0] * pIm_aligned.shape[1]
    good_pixel_count = np.count_nonzero((pIm_aligned.ravel()!=0) &(gIm.ravel()>32))
    return (total_pixel_count - good_pixel_count) / total_pixel_count
    

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
    json_info['scanner_position_origin'] = json_info['scanner_position']
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


def depth_crop_position(xyz_map, cc, xyzd=False):
    """Using the corresponding xyz_map to determine the plot crop position.
    Parameters
    ----------
    xyz_map: nparray
        corresponding gantry coordinates of the pixels on the image, the dim should be m*n*3
        if xyzd is True, then m*n*4
    cc: CoordinateConverter
        plot boundary class
    Returns
    -------
    crop_positions: list
        vertical indices of top of each plot cropping
    """
    # add offsets when reading data
    im_height, im_width, _ = xyz_map.shape
    line_plot_num = np.zeros(im_height)
    y_map = xyz_map[:,:,1].copy()
    x_map = xyz_map[:,:,0].copy()
    y_map[y_map==0] = np.nan
    x_map[x_map==0] = np.nan
    y_line_mean = np.nanmean(y_map, axis=1)
    x_line_mean = np.nanmean(x_map, axis=1)
    
    for i in range(im_height):
        pixel_x = int(im_width / 2)
        pixel_y = i
        # print([pixel_x, pixel_y],'-',[gantry_x, gantry_y])
        if y_line_mean[i] is np.nan or x_line_mean[i] is np.nan:
            if i == 0:
                continue
            else:
                line_plot_num[i] = line_plot_num[i-1]
        plot_row, plot_col = cc.fieldPosition_to_fieldPartition(x_line_mean[i]*0.001, y_line_mean[i]*0.001)
        line_plot_num[i] = cc.fieldPartition_to_plotNum(plot_row, plot_col)
    rle_result = rle(line_plot_num)
    #print(rle_result.astype(int))
    crop_positions = {}
    for rle_element in rle_result:
        plot_num, start_pos, height = rle_element.astype(int)
        if plot_num in crop_positions.keys():
            if height > crop_positions[plot_num][1]:
                crop_positions[plot_num] = [start_pos, height]
        else:
            crop_positions[plot_num] = [start_pos, height]
    return crop_positions
def contour_length(contour):
    c_len = 0
    p_0 = contour[0, :]
    for i in range(1, len(contour)):
        p_1 = contour[i, :]
        seg_len = np.linalg.norm(p_0 - p_1)
        p_0 = p_1
        c_len += seg_len
    return c_len
        
def rle(seq):
    """return the rle encode
    Returns
    -------
    n*3 matrix columns for element, position, length
    """
    i = np.append(np.where(seq[1:] != seq[:-1]), seq.shape[0]-1)
    length = np.diff(np.append(-1, i))
    position = np.cumsum(np.append(0, length))[:-1]
    return np.array([seq[i], position, length]).T

def ply_offset(ply_data, json_info):
    ply_data['vertex']['x'] += json_info['scanner_position_origin'][0]*1000 + json_info['cambox_position'][0]  * 1000
    # ply_data['vertex']['y'] += json_info['scanner_position_origin'][1]*1000 # - json_info['cambox_position'][1] * 1000
    ply_data['vertex']['x'] += 82
    
    if json_info['scan_direction']:
        ply_data['vertex']['y'] += 3450
    else:
        ply_data['vertex']['y'] += 25711
        # ply_data['vertex']['y'] += 3450
    ply_data['vertex']['z'] = ply_data['vertex']['z'] + 3445 - json_info['scanner_position_origin'][2]*1000 + 350
    return ply_data


def heuristic_search_leaf(regions_mask, dxyz, cc, ratio_threshold=3, pixel_lower=0.5, pixel_upper=0.05, max_num_leaf_per_plot=None):
    """ heuristic serach valid leaves from the region mask
    Parameters
    ----------
    regions_mask: ndarray
        candidate regions for finding leaves
    point_cloud_z: ndarray
        pixel height in mm under gantry coordination
    ratio_threshold: int
        ratio threshold of major axis and minor axis
    pixel_lower: float
        relative lower bound of pixel count
    pixel_upper: float
        relative upper bound of pixel count

    Returns
    -------
    leaves_bbox: list
        list of crops position, formatted as [min_row, min_col, max_row, max_col]
    """
    leaves_bbox = []
    label_id_list = []
    regions = regionprops(regions_mask.astype(int), dxyz[:, :,3], coordinates='rc')
    pixel_count_list = [props.area for props in regions]
    # print(len(pixel_count_list))
    pixel_count_list = list(filter(lambda x: x > 20, pixel_count_list))
    trimmed_pixel_count_list = stats.mstats.trim(pixel_count_list, (pixel_lower, pixel_upper),
                                                 relative=True).compressed()
    area_lower = min(trimmed_pixel_count_list)
    area_upper = max(trimmed_pixel_count_list)
    good_regions = []
    good_regions_col_range = []
    im_h, im_w, _ = dxyz.shape
    for props in regions:
        # TODO move mean intensity check on the top combine with region area
        # if  props.area > area_upper or props.area < area_lower:
        #     continue
        if props.mean_intensity == 0:
            continue
        good_pixel_count = np.count_nonzero(props.intensity_image)
        if good_pixel_count / props.area < .99:
            continue
        y0, x0 = props.centroid
        yw, xw = props.weighted_centroid
        if props.major_axis_length < ratio_threshold * props.minor_axis_length:
            continue
        # remove the cutted leaved by the image edge, bleed 1 pxiel
        if 0 in props.bbox or props.bbox[2] >= im_h - 2 or props.bbox[3] >= im_w - 2:
            continue
        # remove region that smaller than 6
        if props.bbox[2] - props.bbox[0] < 6 or props.bbox[3] - props.bbox[1] < 6:
            continue
        good_regions_col_range.append(
            cc.fieldPosition_to_fieldPartition(dxyz[int(y0), int(x0), 1] * 0.001, dxyz[int(y0), int(x0), 2] * 0.001))
        good_regions.append(props)
    # remove components by height
    # 1. for each leaf, find the plot col&range [prev loop]
    # 2. group by col&range
    # 3. fore each group
    #    1. trim by [0.5, 1]
    plots = np.unique(good_regions_col_range, axis=0)
    per_plot_good_regions = []
    for plot in plots:
        good_regions_in_plot = []
        # filter by height
        indice = np.where((good_regions_col_range==plot).all(axis=1))[0]
        regions_in_plot = [good_regions[i] for i in indice]
        region_height_in_plot = [x.max_intensity for x in regions_in_plot]
        min_height = np.min(region_height_in_plot)
        max_height = np.max(region_height_in_plot)
        mid_height = min_height + (max_height - min_height) / 2 
        high_regions_in_plot = [region for region in regions_in_plot if region.max_intensity >= mid_height]
        # filter by trimming of region size per plot 
        pixel_count_list = [props.area for props in high_regions_in_plot]
        area_lower, area_upper = np.percentile(pixel_count_list, [pixel_lower*100, 100 - (pixel_upper*100)])
        good_regions_in_plot = [props for props in high_regions_in_plot if props.area < area_upper and props.area > area_lower]
        if max_num_leaf_per_plot is not None and len(good_regions_in_plot) > max_num_leaf_per_plot:
            good_regions_area_in_plot = [props.area for props in good_regions_in_plot]
            sort_idice = np.argsort(good_regions_area_in_plot)
            target_indice_mid = int(sort_idice.shape[0]/2)
            half_len = int(max_num_leaf_per_plot/2)
            target_indice = sort_idice[target_indice_mid-half_len: target_indice_mid+(max_num_leaf_per_plot-half_len)]
            good_regions_in_plot = [good_regions_in_plot[idx] for idx in target_indice] 
        per_plot_good_regions.extend(good_regions_in_plot)
    good_regions = per_plot_good_regions

    label_id_list = [x.label for x in good_regions]
    leaves_bbox = [x.bbox for x in good_regions]

    return leaves_bbox, label_id_list

def array_zero_to_nan(array):
    nan_array = array.copy().astype(float)
    nan_array[nan_array==0] = np.nan
    return nan_array

def draw_attr(image, attr_dict, loc, line_height, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1, linespace= None):
    if linespace is None:
        linespace = int(line_height/2)
    font_scale = cv2.getFontScaleFromHeight(font, line_height, thickness)
    text_org_x = int(loc[0])
    text_org_y = int(loc[1] - len(attr_dict.keys()) * (line_height + linespace)/2 + (line_height + linespace))
    for key, val in attr_dict.items():
        if val is None:
            val = float('nan')
        text = '{}: {:.2f}'.format(key, val)
        cv2.putText(image, text, (text_org_x, text_org_y), font, font_scale, 255, thickness=thickness)
        text_org_y += (line_height + linespace)

def region_smoothness(image, mask):
    laplace_im = np.abs(filters.laplace(image))
    mask = mask.astype(bool)
    region_area = np.count_nonzero(mask)
    return laplace_im[mask].sum()/region_area