from globus_data_transfer import Transfer
from datetime import date, timedelta
from terra_common import CoordinateConverter as CC
import os, logging, traceback, time, utils, json, argparse
import skimage.io as sio
import pickle
from skimage import measure
from scipy import stats
import numpy as np
from plyfile import PlyData
import midvein_finder
from midvein_finder import SorghumLeafMeasure
import matplotlib.pyplot as plt
import multiprocessing

os.environ['BETYDB_KEY'] = '9999999999999999999999999999999999999999'
REFRESH_TOKEN = 'Ag2QkNl9VBz10O812GmD2gm4O073rmvk3Ow3V0DgkD5wWjnGY4iMUqgaglO8pJWrw00WwbbgWEx1je802BJ1VoyjGWvKN'
PC_TEMP_PATH = os.path.realpath('./terraref/scanner3DTop/tmp/')
LEAF_LEN_RESULT_PATH = './leaf_len_result'

def options():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def connected_compoent(image, gradient_threshold=10):
    '''
    if the upper and lower bound is not given, throw the regions that  pixel count from top and bottom 1/4
    '''
    
    gx, gy = np.gradient(image)
    edge = (np.abs(gx)<gradient_threshold) & (np.abs(gy)<gradient_threshold) # & (pc_xyz_square[:, :, 2]>1)
    #all_labels = measure.label(edge, background=0)
    all_labels = measure.label(edge, connectivity=1)
    label_img = np.zeros(all_labels.shape)
    return all_labels


def get_pc_data_from_globus(folder_name, logger, sensor_name='east'):
    '''
    saved as 
    '''
    globus_root_path = os.path.join('/ua-mac/Level_1/scanner3DTop/', folder_name[:10], folder_name)
    target_path = os.path.join(PC_TEMP_PATH, folder_name+'.ply')
    t = Transfer('Terraref', 'Lab Server', transfer_rt=REFRESH_TOKEN)
    # print(globus_root_path)
    for file in t.ls_src_dir(globus_root_path):
        # print(file['name'])
        if sensor_name in file['name']:
            globus_file_path = os.path.join(globus_root_path, file['name'])
            break
    response = t.transfer_file(globus_file_path, target_path)
    logger.info('transfer task of {} submitted, response message:{}, task_id:{}.'
          .format(folder_name, response['message'], response['task_id']))
    # block to check status
    task_status = None
    wait_count = 0
    while not t.transferClient.task_wait(response['task_id'], timeout=60):
        wait_count += 1
        if wait_count > 20:
            logger.warning('*** waited task finish for 20 minutes')
            wait_count = 0
        pass
    logger.info('transfer task of {} finished, task_id:{}.'
          .format(folder_name, response['task_id']))
    return t.get_task(response['task_id'])


def crop_mask_by_id(mask, xyz_map, mask_id):
    mask = mask==mask_id
    mask = mask.astype(int)
    # remove zero cols and rows
    cropped_mask = mask.copy()
    cropped_mask = cropped_mask[:,~np.all(mask == 0, axis=0)]
    cropped_mask = cropped_mask[~np.all(mask == 0, axis=1)]
    cropped_xyz_map = xyz_map.copy()
    cropped_xyz_map = cropped_xyz_map[:,~np.all(mask == 0, axis=0),:]
    cropped_xyz_map = cropped_xyz_map[~np.all(mask == 0, axis=1), :, :]
    # and 1 pixel boarder
    cropped_mask = np.pad(cropped_mask, [(1, 1), (1, 1)], mode='constant')
    cropped_xyz_map = np.asarray([np.pad(xyz, [(1, 1), (1, 1)], mode='edge') for xyz in cropped_xyz_map.transpose(2,0,1)])
    cropped_xyz_map = cropped_xyz_map.transpose(1,2,0)
    cropped_position = [[np.where(~np.all(mask == 0, axis=1))[0][0],np.where(~np.all(mask == 0, axis=1))[0][-1]],
                       [np.where(~np.all(mask == 0, axis=0))[0][0],np.where(~np.all(mask == 0, axis=0))[0][-1]]]
    return cropped_mask, cropped_xyz_map, cropped_position


def find_leaves(ply_depth_map):
    # downsampling
    ply_depth_map = ply_depth_map[::4, ::4, :]
    # connected component
    cc_mask = connected_compoent(ply_depth_map[:,:,1:])
    cc_mask = np.nan_to_num(cc_mask).astype(int)
    # heuristic search leaves that ellipse major > 3 * minor
    reg_props = measure.regionprops(cc_mask)
    good_leaves_id_list = []
    for reg_prop in reg_props:
        if reg_prop.major_axis_length < 4 * reg_prop.minor_axis_length:
            continue
        good_leaves_id_list.append(reg_prop.label)
    mask_list = []
    xyzd_list = []
    crop_position_list = []
    for i in good_leaves_id_list:
        id_mask, id_xyzd, id_crop_pos = crop_mask_by_id(cc_mask, ply_depth_map, i)
        mask_list.append(id_mask)
        xyzd_list.append(id_xyzd)
        crop_position_list.append(id_crop_pos)
    return mask_list, xyzd_list, crop_position_list


def run_analysis(data_folder, log_lv=logging.DEBUG, per_plot=True):
    # TODO better logging
    data_folder = os.path.join(data_folder, '')
    folder_name = os.path.basename(os.path.dirname(data_folder))
    pkl_file_path = os.path.join(LEAF_LEN_RESULT_PATH,folder_name+'.pkl')
    cpname = multiprocessing.current_process().name
    
    # Logger
    logger = logging.getLogger('ppln_'+folder_name+'_'+cpname)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s %(levelname)s:\t%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Start processing')
    
    # Skip processed folders
    if os.path.isfile(pkl_file_path):
        try:
            with open(pkl_file_path, 'rb') as f:
                data = pickle.load(f)
                if 'leaf_len' and 'plot_id' in data.keys():
                    logger.info('Pkl exist, skip.')
                    return 1
        except Exception as e:
            logger.warning('.pkl file exist but corrupted.')
            
    # Get data
    # Get .png
    filename_list = os.listdir(data_folder)
    for filename in filename_list:
        if 'east_0_g.png' in filename:
            gIm_name = filename
        if 'east_0_p.png' in filename:
            pIm_name = filename
        if 'metadata.json' in filename:
            json_name = filename
    try:
        gIm = sio.imread(os.path.join(data_folder, gIm_name))
        pIm = sio.imread(os.path.join(data_folder, pIm_name))
    except:
        logger.error('Image reading error! Skip.')
        return -1
    # Get .ply
    if not os.path.isfile(os.path.expanduser(os.path.join(PC_TEMP_PATH, folder_name+'.ply'))):
        try:
            get_pc_data_from_globus(folder_name, logger)
        except:
            tb = traceback.format_exc()
            logger.error('Download from globus error!\n'+tb)
            return -2
    ply_data_path = os.path.expanduser(os.path.join(PC_TEMP_PATH, folder_name+'.ply'))
    try:
        ply_data = PlyData.read(ply_data_path)
    except Exception as e:
        logger.error('ply file reading error! Skip.'.format(folder_name+'.ply'))
        os.remove(ply_data_path)
        return -1
    # Read json file
    try:
        with open(os.path.join(data_folder, json_name), 'r') as json_f:
            json_data = json.load(json_f)
        json_info = utils.get_json_info(json_data)
    except Exception as e:
        logger.error('Load json unsuccessful.')
        return -4
    if gIm.shape != (21831, 2048) or pIm.shape != (21831, 2050):
        logger.error('Image dim does not match. Excepted for pIm:{} gIm:{}; but got pIm:{}, gIm:{}. Skip.'.format((21831, 2050), (21831, 2048), pIm.shape, gIm.shape))
        os.remove(ply_data_path)
        return -3
    
    # bety query
    cc = CC(useSubplot=True)
    logger.info('bety query start.')
    cc.bety_query(json_info['date'].strftime('%Y-%m-%d'), useSubplot=True)
    logger.info('bety query complete.')
    
    # align pointcloud 
    ply_xyz_map = utils.ply2xyz(ply_data, pIm, gIm)
    ply_depth = np.dstack([pIm[:, 2:], ply_xyz_map])

    if not per_plot:
        pass

    # crop
    if json_info['scan_direction']:
        vertical_top_list = np.array(list(range(0, 21830-770, 770)))
    else:
        vertical_top_list = np.array(list(range(360-320, 21830-770, 770)))
    xyzd_slice_list = []
    for vertical_top in vertical_top_list:
        xyzd_slice_list.append(ply_depth[vertical_top: vertical_top+770, :, :])
        
    # get plot id w/ module terra_common
    plot_id_list = []
    for vertical_top in vertical_top_list:
        if json_info['scan_direction']:
            centroid = [vertical_top+770/2, 1025]
        else:
            centroid = [-(vertical_top+770/2), 1025]
        plot_id = cc.pixel_to_plotNum(centroid[1], centroid[0], json_info['scanner_position'], 
                                      json_info['fov'], json_info['scan_distance'], 
                                      pIm.shape[1], pIm.shape[0])
        plot_id_list.append(plot_id)
        
    # for each plot
    leaf_length_list = []
    leaf_width_list = []
    for xyzd_slice in xyzd_slice_list:
        mask_list, xyzd_list, crop_position_list = find_leaves(xyzd_slice)
        plot_leaf_length_list = []
        plot_leaf_width_list = []
        for leaf_mask, leaf_xyzd, leaf_crop_pos in zip(mask_list, xyzd_list, crop_position_list):
            slm = SorghumLeafMeasure(leaf_mask, leaf_xyzd[:, :, :3])
            slm.calc_leaf_length()
            slm.calc_leaf_width()
            plot_leaf_length_list.append(slm.leaf_len)
            plot_leaf_width_list.append(slm.leaf_len)
            
        # average mid 1/2 data
        avg_leaf_length = stats.trim_mean(plot_leaf_length_list, 0.25)
        avg_leaf_width = stats.trim_mean(plot_leaf_width_list, 0.25)
        # print(avg_leaf_len)
        leaf_length_list.append(avg_leaf_length)
        leaf_width_list.append(avg_leaf_width)
    leaf_len_dict = {}
    leaf_len_dict['leaf_length'] = leaf_length_list
    leaf_len_dict['leaf_width'] = leaf_length_list
    leaf_len_dict['plot_id'] = plot_id_list
    # write one scan into a file
    with open(pkl_file_path, 'wb') as pickle_f:
        pickle.dump(leaf_len_dict, pickle_f)
    logger.info('finished')
    return 0
