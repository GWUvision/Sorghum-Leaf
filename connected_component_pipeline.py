from globus_data_transfer import Transfer
from datetime import date, timedelta
from terra_common import CoordinateConverter as CC
import os, logging, traceback, time, utils, json, argparse, shutil
import skimage.io as sio
import pickle
from skimage import measure
from scipy import stats
import numpy as np
from plyfile import PlyData
from midvein_finder import SorghumLeafMeasure
import multiprocessing
from timeit import default_timer as timer

os.environ['BETYDB_KEY'] = '9999999999999999999999999999999999999999'
REFRESH_TOKEN = 'Ag2QkNl9VBz10O812GmD2gm4O073rmvk3Ow3V0DgkD5wWjnGY4iMUqgaglO8pJWrw00WwbbgWEx1je802BJ1VoyjGWvKN'
PC_TEMP_PATH = os.path.realpath('./terraref/scanner3DTop/tmp/')
LEAF_LEN_RESULT_PATH = './leaf_len_result'

def options():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def connected_component(image, gradient_threshold=10):
    '''
    if the upper and lower bound is not given, throw the regions that  pixel count from top and bottom 1/4
    '''
    
    gx, gy = np.gradient(image)
    edge = (np.abs(gx) < gradient_threshold) & (np.abs(gy) < gradient_threshold)  # & (pc_xyz_square[:, :, 2]>1)
    #all_labels = measure.label(edge, background=0)
    all_labels = measure.label(edge, connectivity=1)
    label_img = np.zeros(all_labels.shape)
    return all_labels


def get_ply_data_from_globus(target_folder, logger=None, sensor_name='east'):
    '''
    saved as 
    '''
    # TODO optional refresh token
    target_folder = os.path.join(target_folder, '')
    folder_name = os.path.basename(os.path.dirname(target_folder))
    globus_root_path = os.path.join('/ua-mac/Level_1/scanner3DTop/', folder_name[:10], folder_name)
    target_path = os.path.join(target_folder, folder_name + '_' + sensor_name + '.ply')
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
    mask = (mask == mask_id)
    mask = mask.astype(int)
    # remove zero cols and rows
    cropped_mask = mask.copy()
    cropped_mask = cropped_mask[:, ~np.all(mask == 0, axis=0)]
    cropped_mask = cropped_mask[~np.all(mask == 0, axis=1)]
    cropped_xyz_map = xyz_map.copy()
    cropped_xyz_map = cropped_xyz_map[:, ~np.all(mask == 0, axis=0), :]
    cropped_xyz_map = cropped_xyz_map[~np.all(mask == 0, axis=1), :, :]
    # and 1 pixel boarder
    cropped_mask = np.pad(cropped_mask, [(1, 1), (1, 1)], mode='constant')
    cropped_xyz_map = np.asarray([np.pad(xyz, [(1, 1), (1, 1)], mode='edge') for xyz in cropped_xyz_map.transpose(2,0,1)])
    cropped_xyz_map = cropped_xyz_map.transpose(1, 2, 0)
    cropped_position = [[np.where(~np.all(mask == 0, axis=1))[0][0], np.where(~np.all(mask == 0, axis=1))[0][-1]],
                       [np.where(~np.all(mask == 0, axis=0))[0][0], np.where(~np.all(mask == 0, axis=0))[0][-1]]]
    return cropped_mask, cropped_xyz_map, cropped_position


def find_leaves(dxyz_map):
    # downsampling
    dxyz_map = dxyz_map[::4, ::4, :]
    # connected component
    cc_mask = connected_component(dxyz_map[:, :, 3])
    # heuristic search leaves that ellipse major > 3 * minor
    leaf_crops, leaf_bbox, label_id_list = utils.heuristic_search_leaf(cc_mask, dxyz_map[:, :, 3])
    mask_list = []
    xyzd_list = []
    crop_position_list = []
    for i in label_id_list:
        id_mask, id_xyzd, id_crop_pos = crop_mask_by_id(cc_mask, dxyz_map, i)
        mask_list.append(id_mask)
        xyzd_list.append(id_xyzd)
        crop_position_list.append(id_crop_pos)
    return mask_list, xyzd_list, crop_position_list


def run_analysis(raw_data_folder, ply_data_folder, output_folder,
                 sensor_name='east', download_ply=False, per_plot=True, log_lv=logging.INFO):
    '''
    Run single analysis
    :param raw_data_folder: folder to the raw data. Eg. /path/to/data/raw/scanner3DTop/2016-04-30/2016-04-30__12-55-42-331/
    :param ply_data_folder: folder to the ply data. Eg. /path/to/data/Level_1/scanner3DTop/2016-04-30/2016-04-30__12-55-42-331/
    :param output_folder: save the middle output for one scan Eg. /path/to/output/
    :param sensor_name: east or west
    :param download_ply: bool
    :param per_plot: bool
    :param log_lv: default is Debug
    :return: status code
    '''
    init_start = timer()
    # TODO better logging
    # Logger
    cpname = multiprocessing.current_process().name
    logger = logging.getLogger('ppln_' + os.path.basename(os.path.dirname(raw_data_folder)) + '_' + cpname)
    logger.setLevel(log_lv)
    formatter = logging.Formatter('%(asctime)s - %(name)s %(levelname)s:\t%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(log_lv)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Start processing')

    raw_data_folder = os.path.join(raw_data_folder, '')
    ply_data_folder = os.path.join(ply_data_folder, '')
    folder_name = os.path.basename(os.path.dirname(raw_data_folder))
    pkl_file_path = os.path.join(output_folder, folder_name + '_' + sensor_name + '.pkl')
    # get p/g image and ply filename
    for filename in os.listdir(raw_data_folder):
        if sensor_name + '_0_g.png' in filename:
            gIm_name = filename
        if sensor_name + '_0_p.png' in filename:
            pIm_name = filename
        if 'metadata.json' in filename:
            json_name = filename
    if download_ply:
        ply_data_path = os.path.expanduser(os.path.join(ply_data_folder, folder_name + '_' + sensor_name + '.ply'))
    else:
        # check ply existence
        ply_data_path = None
        for filename in os.listdir(ply_data_folder):
            if sensor_name in filename:
                ply_data_path = os.path.expanduser(os.path.join(ply_data_folder, filename))
        if ply_data_path is None:
            logger.error('ply file does not exist. sensor:{}, path:{}'.format(sensor_name, ply_data_folder))
            return -1
        pass

    # Skip processed folders
    if os.path.isfile(pkl_file_path):
        try:
            with open(pkl_file_path, 'rb') as f:
                data = pickle.load(f)
                if 'leaf_length' and 'plot_id' in data.keys():
                    logger.info('Pkl exist, skip.')
                    return 1
        except Exception as e:
            logger.warning('.pkl file exist but corrupted.')
    init_end = timer()
    logger.debug('initialization time elapsed: {0:.3f}s'.format(init_end-init_start))
    # Get data
    get_data_start = timer()
    # Get .png
    try:
        gIm = sio.imread(os.path.join(raw_data_folder, gIm_name))
        pIm = sio.imread(os.path.join(raw_data_folder, pIm_name))
    except:
        logger.error('Image reading error! Skip.')
        return -1
    # Get .ply
    if download_ply:
        # TODO check ply integrity before
        if not os.path.isfile(ply_data_path):
            try:
                get_ply_data_from_globus(ply_data_folder, logger, sensor_name=sensor_name)
            except:
                tb = traceback.format_exc()
                logger.error('Download from globus error!\n'+tb)
                return -2
    # read .ply
    try:
        ply_data = PlyData.read(ply_data_path)
    except Exception as e:
        logger.error('ply file reading error! Skip. file_path:{}'.format(ply_data_path))
        if download_ply:
            shutil.rmtree(ply_data_folder)
        return -1
    # Read json file
    try:
        with open(os.path.join(raw_data_folder, json_name), 'r') as json_f:
            json_data = json.load(json_f)
        json_info = utils.get_json_info(json_data, sensor=sensor_name)
    except Exception as e:
        logger.error('Load json file unsuccessful.')
        return -4
    if gIm.shape != (21831, 2048) or pIm.shape != (21831, 2050):
        logger.error('Image dim does not match. Excepted for pIm:{} gIm:{}; but got pIm:{}, gIm:{}. Skip.'.format((21831, 2050), (21831, 2048), pIm.shape, gIm.shape))
        if download_ply:
            shutil.rmtree(ply_data_folder)
        return -3
    get_data_end = timer()
    logger.debug('reading data time elapsed: {0:.3f}s'.format(get_data_end - get_data_start))
    # add offset to ply
    apply_offset_start = timer()
    ply_data = utils.ply_offset(ply_data, json_info)
    apply_offset_end = timer()
    logger.debug('ply apply offset time elapsed: {0:.3f}s'.format(apply_offset_end - apply_offset_start))
    # bety query
    bety_query_start = timer()
    cc = CC(useSubplot=True)
    logger.info('bety query start.')
    cc.bety_query(json_info['date'].strftime('%Y-%m-%d'), useSubplot=True)
    logger.info('bety query complete.')
    bety_query_end = timer()
    logger.debug('bety query time elapsed: {0:.3f}s'.format(bety_query_end - bety_query_start))
    # align pointcloud
    align_start = timer()
    ply_xyz_map = utils.ply2xyz(ply_data, pIm, gIm)
    ply_dxyz = np.dstack([pIm[:, 2:], ply_xyz_map])
    align_end = timer()
    logger.debug('align data time elapsed: {0:.3f}s'.format(align_end - align_start))
    if not per_plot:
        pass

    # crop
    crop_start = timer()
    dxyz_slice_list = []
    plot_id_list = []
    crop_position_dict = utils.depth_crop_position(ply_xyz_map, cc)
    for plot_id in crop_position_dict.keys():
        start_pos, height = crop_position_dict[plot_id]
        end_pos = start_pos + height
        plot_id_list.append(plot_id)
        dxyz_slice_list.append(ply_dxyz[start_pos: end_pos, :, :])
    crop_end = timer()
    logger.debug('crop data time elapsed: {0:.3f}s'.format(crop_end - crop_start))
    # for each plot
    leaf_length_list = []
    leaf_width_list = []
    for dxyz_slice in dxyz_slice_list:
        slice_start = timer()
        mask_list, dxyz_list, crop_position_list = find_leaves(dxyz_slice)
        plot_leaf_length_list = []
        plot_leaf_width_list = []
        for leaf_mask, leaf_dxyz, leaf_crop_pos in zip(mask_list, dxyz_list, crop_position_list):
            leaf_start = timer()
            slm = SorghumLeafMeasure(leaf_mask, leaf_dxyz[:, :, 1:])
            leaf_length_start = timer()
            slm.calc_leaf_length()
            leaf_width_start = timer()
            slm.calc_leaf_width()
            leaf_end = timer()
            logger.debug('leaf processing time elapsed: total {0:.3f} s\n'
                         '\tinit: {0:.3f} s \n\tlength: {0:.3f} s \n\twidth: {0:.3f} s'
                         .format(leaf_end - leaf_start,
                                 leaf_length_start - leaf_start,
                                 leaf_width_start - leaf_length_start,
                                 leaf_end - leaf_width_start))
            plot_leaf_length_list.append(slm.leaf_len)
            plot_leaf_width_list.append(slm.leaf_len)
            
        # average mid 1/2 data
        avg_leaf_length = stats.trim_mean(plot_leaf_length_list, 0.25)
        avg_leaf_width = stats.trim_mean(plot_leaf_width_list, 0.25)
        # print(avg_leaf_len)
        leaf_length_list.append(avg_leaf_length)
        leaf_width_list.append(avg_leaf_width)
        slice_end = timer()
        logger.debug('slice processing total time elapsed: {0:.3f}s'.format(slice_end - slice_start))
    leaf_len_dict = {}
    leaf_len_dict['leaf_length'] = leaf_length_list
    leaf_len_dict['leaf_width'] = leaf_length_list
    leaf_len_dict['plot_id'] = plot_id_list
    # write one scan into a file
    with open(pkl_file_path, 'wb') as pickle_f:
        pickle.dump(leaf_len_dict, pickle_f)
    if download_ply:
        shutil.rmtree(ply_data_folder)
    logger.info('finished')
    return 0
