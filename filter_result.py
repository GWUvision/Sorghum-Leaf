import os
import pickle
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm

def kalman( x, P, m, R, motion, Q, F, H ):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position ( same shape as H * x )
    R: measurement noise ( same shape as H )
    motion: external motion added to state vector x
    Q: motion noise ( same shape as P )
    F: next state function: x_prime = F * x
    H: measurement function: position = H * x

    Return: the updated and predicted new values for ( x, P )

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H 
    '''
    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    if np.isnan(m).any():
        y = 0
    else:
        y = m - H * x


    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain

    x = x + K * y
    I = np.matrix( np.eye( F.shape[0] ) ) # identity matrix
    P = ( I - K * H ) * P
    # PREDICT x, P based on motion
    x = F * x + motion
    P = F * P * F.T + Q
    return x, P

# TODO refactor to reuse the kalman filter function
def kalman_leaf_length(day_array, measure_array, assumed_start_val=100, assumed_start_velocity=5,
                      P=None, Q=None):
    for i in range(max(day_array)):
        if i not in day_array:
            day_array = np.append(day_array, i)
            measure_array = np.append(measure_array, np.nan)
    measure_array = measure_array[np.argsort(day_array)]
    day_array.sort()
    N = max(day_array)
    x = np.matrix( [assumed_start_val, assumed_start_velocity] ).T 
    if P is None:
        P =  np.matrix( '''
            1000. 0. ;
            0. 100.''' )
    R = 10000
    
    F = np.matrix( '''
        1. 1. ;
        0. 1. ''' )
    H = np.matrix( '''
        1. 0. ''' )
    motion = np.matrix( '0. 0.' ).T
    if Q is None:
        Q =  np.matrix( '''
            25. 0. ;
            0. 4. ''' )
    m = np.matrix( '0.' ).T
    kalman_x = np.zeros( N )
    for n in range(N):
        m[0] = measure_array[n]
        x, P = kalman( x, P, m, R, motion, Q, F, H )
        kalman_x[n] = x[0]
    return day_array, kalman_x

def kalman_leaf_width(day_array, measure_array, assumed_start_val=15.0, assumed_start_velocity=0.8,
                      P=None, Q=None):
    for i in range(max(day_array)):
        if i not in day_array:
            day_array = np.append(day_array, i)
            measure_array = np.append(measure_array, np.nan)
    measure_array = measure_array[np.argsort(day_array)]
    day_array.sort()
    N = max(day_array)
    x = np.matrix( [assumed_start_val, assumed_start_velocity] ).T 
    if P is None:
        P =  np.matrix( '''
            200. 0. ;
            0. 10.''' )
    R = 1000
    
    F = np.matrix( '''
        1. 1. ;
        0. 1. ''' )
    H = np.matrix( '''
        1. 0. ''' )
    motion = np.matrix( '0. 0.' ).T
    if Q is None:
        Q =  np.matrix( '''
            16. 0. ;
            0. 2. ''' )
    m = np.matrix( '0.' ).T
    kalman_x = np.zeros( N )
    for n in range(N):
        m[0] = measure_array[n]
        x, P = kalman( x, P, m, R, motion, Q, F, H )
        kalman_x[n] = x[0]
    return day_array, kalman_x

def leaf_data_dict_to_dataframe(data_dict):
    result_list = []
    for (date, sensor), plot_dict in tqdm(data_dict.items(), desc='parse dict to list'):
        for (plot_col, plot_range), leaves_list in plot_dict.items():
            for leaf_length, leaf_width in leaves_list:
                result_list.append([date, sensor, plot_col, plot_range, leaf_length, leaf_width])
    leaves_df = pd.DataFrame(result_list)
    leaves_df.columns=['date', 'sensor', 'col', 'range', 'leaf_length', 'leaf_width']
    leaves_df['date']=leaves_df['date'].astype('datetime64[ns]')
    return leaves_df

def subplot_col_range_to_site_str(subplot_col, subplot_range, season_num):
    plot_range = subplot_range
    plot_col = int((subplot_col + 1) / 2)
    if (subplot_col % 2) != 0:
        subplot = 'W'
    else:
        subplot = 'E'
    site_str = 'MAC Field Scanner Season {} Range {} Column {} {}'\
        .format(str(season_num), str(plot_range), str(plot_col), subplot)
    return site_str

def remove_outliars(df, by=None, threshold=1):
    if by is None:
        by = df.columns
    if (not hasattr(arg, "strip") and
            hasattr(arg, "__iteritems__") or
            hasattr(arg, "__iter__")):
        by = list(by)
    else:
        by = [by, ]
    if len(by) > 1:
        return df[np.abs((stats.zscore(df[by])) < threshold).all(axis=1)]
    else:
        return df[np.abs(stats.zscore(df[by])) < threshold]

if __name__ == '__main__':
    data_file_path = './leaves.pkl'
    season_num = 4
    pkl_save_folder = './s4_pkl_result'
    # read data
    with open(data_file_path, 'rb') as f:
        leaves_dict = pickle.load(f)
    leaves_df = leaf_data_dict_to_dataframe(leaves_dict)
    # remove NaN
    leaves_df.loc[(leaves_df[['leaf_width']] == 0).all(axis=1), 'leaf_width'] = np.nan
    leaves_df = leaves_df.dropna()
    # remove outliers of leaf width
    leaves_df.loc[(np.abs(stats.zscore(leaves_df[['leaf_width']])) > 0.1).all(axis=1), 'leaf_width'] = np.nan
    leaves_df = leaves_df.dropna()
    leaves_df.loc[(np.abs(stats.zscore(leaves_df[['leaf_width']])) > 4).all(axis=1), 'leaf_width'] = np.nan
    leaves_df = leaves_df.dropna()
    # find all avaliable plot col-range
    col_range = np.unique(leaves_df[['col', 'range']].values, axis=0)
    # remove row with zeros
    col_range = col_range[np.all(col_range!=0, axis=1), :]
    all_plot_result_df = pd.DataFrame()
    for plot_col, plot_range in tqdm(col_range, desc='kalman filtering'):
        # select plot
        selected_plot_df = leaves_df[(leaves_df['col']==plot_col) & (leaves_df['range']==plot_range)]
        # TODO could move two filter out of for loop for better readiability 
        # filtered if only few days have data
        if len(selected_plot_df['date'].unique()) < 30:
            continue
        # filter days with few data points 
        selected_plot_df = selected_plot_df.groupby(['date', 'col', 'range'])\
            .filter(lambda x: x['leaf_length'].count() > 4)
        if len(selected_plot_df['date'].unique()) < 30:
            continue
        # mean by average all (west east together)
        selected_plot_df = selected_plot_df.groupby(['date']).mean().reset_index()
        # transfer the data into discrete time_array, measure_array format
        # add delta_day column
        first_day = selected_plot_df['date'].min()
        selected_plot_df['delta_days'] = np.nan
        selected_plot_df['delta_days'] = (selected_plot_df['date'] - first_day)
        selected_plot_df['delta_days'] = selected_plot_df['delta_days'].apply(lambda delta_t: delta_t.days)
        length_measure_array = selected_plot_df['leaf_length'].values
        width_measure_array = selected_plot_df['leaf_width'].values
        time_array = selected_plot_df['delta_days'].values
        # put it into kalman filter
        result_day_array, result_kalman_length = kalman_leaf_length(time_array, length_measure_array)
        result_day_array, result_kalman_width = kalman_leaf_width(time_array, width_measure_array)
        # select from first day to the day with largest measurement
        max_loc = np.argmax(result_kalman_length)
        result_day_array = result_day_array[:max_loc]
        result_kalman_length = result_kalman_length[:max_loc]
        result_kalman_width = result_kalman_width[:max_loc]
        result_df = pd.DataFrame(columns=['local_datetime', 'leaf_length', 'leaf_width', 'delta_days', 'site']) 
        result_df['delta_days'] = result_day_array
        result_df['delta_days'] = pd.to_timedelta(result_df['delta_days'], unit='d')
        result_df['leaf_length'] = result_kalman_length
        result_df['leaf_width'] = result_kalman_width
        result_df['local_datetime'] = first_day
        result_df['local_datetime'] = result_df['local_datetime'] + result_df['delta_days']
        result_df['site'] = subplot_col_range_to_site_str(plot_col, plot_range, season_num)
        result_df['species'] = 'Sorghum bicolor'
        result_df['citation_author'] = 'Zeyu Zhang'
        result_df['citation_year'] = '2019'
        result_df['citation_title'] = 'Maricopa Field Station Data and Metadata'
        result_df['method'] = 'Scanner 3d ply data to leaf length'
        result_df = result_df.drop(['delta_days'], axis=1)
        
        # save result only
        result_df.to_pickle(os.path.join(pkl_save_folder,'col_{}_range_{}.pkl'.format(plot_col, plot_range)))
        all_plot_result_df = all_plot_result_df.append(result_df)
    all_plot_result_df.to_csv('./betydb_s{}_length_result.csv'.format(season_num), index=False, float_format='%.3f')
    
    
        
