import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import os

sst_data = {}

def is_leap_year(year):

    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def open_sst_dataset(data_directory, start_year, end_year, center_lat, center_lon, lat_span, lon_span, sampling_window):

    global sst_data 

    lat_resolution = 0.25
    lon_resolution = 0.25

    extended_lat_min = center_lat - lat_span - sampling_window
    extended_lat_max = center_lat + lat_span + sampling_window
    extended_lon_min = center_lon - lon_span - sampling_window
    extended_lon_max = center_lon + lon_span + sampling_window

    for year in range(start_year, end_year + 1):
        file_path = os.path.join(data_directory, f'sst.day.mean.{year}.nc')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with nc.Dataset(file_path) as dataset:
            latitudes = dataset.variables['lat'][:]
            longitudes = dataset.variables['lon'][:]

            time_var = dataset.variables['time']
            time_units = time_var.units
            calendar = getattr(time_var, 'calendar', 'gregorian')
            times = nc.num2date(time_var[:], units=time_units, calendar=calendar)

            lat_start_idx = int((extended_lat_min - latitudes.min()) / lat_resolution)
            lat_end_idx = int((extended_lat_max - latitudes.min()) / lat_resolution) + 1
            lon_start_idx = int((extended_lon_min - longitudes.min()) / lon_resolution)
            lon_end_idx = int((extended_lon_max - longitudes.min()) / lon_resolution) + 1

            lat_start_idx = max(lat_start_idx, 0)
            lat_end_idx = min(lat_end_idx, len(latitudes))
            lon_start_idx = max(lon_start_idx, 0)
            lon_end_idx = min(lon_end_idx, len(longitudes))

            lat_subset = latitudes[lat_start_idx:lat_end_idx]
            lon_subset = longitudes[lon_start_idx:lon_end_idx]

            sst_var = dataset.variables['sst'][:, lat_start_idx:lat_end_idx, lon_start_idx:lon_end_idx] 


            fill_value = getattr(dataset.variables['sst'], '_FillValue', None)
            if fill_value is not None:
                sst_var = np.where(sst_var == fill_value, np.nan, sst_var)
            sst_var = np.where(sst_var < -1e+30, np.nan, sst_var)

            sst_data[year] = {
                'sst': sst_var, 
                'latitudes': lat_subset, 
                'longitudes': lon_subset,
                'times': times, 
                'days_in_year': 366 if is_leap_year(year) else 365
            }

def get_day_of_year(sample_date):

    return sample_date.timetuple().tm_yday - 1 

def get_sst_data(year, sample_date):

    global sst_data  
    if year not in sst_data:
        print(f"Year {year} not found in sst_data")
        return None

    sst_var = sst_data[year]['sst']
    times = sst_data[year]['times']

    try:
        sample_date_obj = datetime.strptime(sample_date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid sample_date format: {sample_date}")
        return None

    try:
        sample_num = np.where(times == sample_date_obj)[0][0]
    except IndexError:
        print(f"sample_date {sample_date} not found in year {year}")
        return None
    return sst_var[sample_num] 

def extract_center_point_temperature(center_lat, center_lon, t_train, L, t_gap):

    global sst_data  
    try:
        t_datetime = datetime.strptime(str(t_train), "%Y%m%d")
    except ValueError:
        print(f"Invalid t_train format in extract_center_point_temperature: {t_train}")
        return np.array([])

    sst_t = []

    for i in range(L):
        sample_date = t_datetime + timedelta(days=i * t_gap)
        sample_year = sample_date.year
        sample_date_str = sample_date.strftime("%Y-%m-%d")

        sst_data_year = get_sst_data(sample_year, sample_date_str)
        if sst_data_year is None:
            print(f"SST data missing for center point at date: {sample_date_str}")
            sst_t.append(np.nan)
            continue

        latitudes = sst_data[sample_year]['latitudes']
        longitudes = sst_data[sample_year]['longitudes']

        lat_resolution = 0.25
        lon_resolution = 0.25

        lat_start = latitudes.min()
        lon_start = longitudes.min()

        lat_index = int(round((center_lat - lat_start) / lat_resolution))
        lon_index = int(round((center_lon - lon_start) / lon_resolution))

        if lat_index < 0 or lat_index >= len(latitudes):
            print(f"lat_index {lat_index} out of range for year {sample_year}")
            sst_t.append(np.nan)
            continue
        if lon_index < 0 or lon_index >= len(longitudes):
            print(f"lon_index {lon_index} out of range for year {sample_year}")
            sst_t.append(np.nan)
            continue

        try:
            sst_data_value = sst_data_year[lat_index, lon_index]
        except IndexError:
            print(f"IndexError accessing SST data at date: {sample_date_str}, lat_index: {lat_index}, lon_index: {lon_index}")
            sst_t.append(np.nan)
            continue

        if sst_data_value < -1e+30:
            sst_data_value = np.nan
        sst_t.append(sst_data_value)

    return np.array(sst_t)


def extract_region_temperature(center_lat, center_lon, lat_span, lon_span, t_train, M, t_gap):

    global sst_data 
    try:
        t_datetime = datetime.strptime(str(t_train), "%Y%m%d")
    except ValueError:
        print(f"Invalid t_train format: {t_train}")
        return np.array([])

    region_data = []

    for i in range(M):
        sample_date = t_datetime + timedelta(days=i * t_gap)
        sample_year = sample_date.year
        sample_date_str = sample_date.strftime("%Y-%m-%d")

        sst_data_year = get_sst_data(sample_year, sample_date_str)
        if sst_data_year is None:
            print(f"SST data missing for date: {sample_date_str}")
            latitudes = sst_data.get(sample_year, {}).get('latitudes', [])
            longitudes = sst_data.get(sample_year, {}).get('longitudes', [])
            if len(latitudes) == 0 or len(longitudes) == 0:
                flattened_slice = np.array([])
            else:
                num_points = len(latitudes) * len(longitudes)
                flattened_slice = np.full(num_points, np.nan)
            region_data.append(flattened_slice)
            continue

        latitudes = sst_data[sample_year]['latitudes']
        longitudes = sst_data[sample_year]['longitudes']

        lat_min = center_lat - lat_span
        lat_max = center_lat + lat_span
        lon_min = center_lon - lon_span
        lon_max = center_lon + lon_span

        lat_resolution = 0.25
        lon_resolution = 0.25

        lat_start_idx = int(round((lat_min - latitudes.min()) / lat_resolution))
        lat_end_idx = int(round((lat_max - latitudes.min()) / lat_resolution)) + 1
        lon_start_idx = int(round((lon_min - longitudes.min()) / lon_resolution))
        lon_end_idx = int(round((lon_max - longitudes.min()) / lon_resolution)) + 1

        lat_start_idx = max(lat_start_idx, 0)
        lat_end_idx = min(lat_end_idx, len(latitudes))
        lon_start_idx = max(lon_start_idx, 0)
        lon_end_idx = min(lon_end_idx, len(longitudes))

        sst_slice = sst_data_year[lat_start_idx:lat_end_idx, lon_start_idx:lon_end_idx]

        sst_slice = np.where(sst_slice < -1e+30, np.nan, sst_slice)

        flattened_slice = sst_slice.flatten(order='C')  # 从南到北，从西到东

        region_data.append(flattened_slice)

    region_data = np.array(region_data)

    return region_data

def clean_nan_data(data):
    if data.size == 0:
        return data
    
    col_mean = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    if col_mean.size != 0:
        data[inds] = np.take(col_mean, inds[1])
    return data

def construct_delay_embedding_matrix(data, L, M):

    if len(data) < L + M:
        print("Not enough data to construct delay embedding matrix")
        return np.array([])
    delay_embedding_matrix = np.array([data[i:i + M] for i in range(L)])
    return delay_embedding_matrix

def process_last_column(batch_labels):

    if batch_labels.ndim != 3:
        print("batch_labels is not a 3D array")
        return np.array([])

    last_columns = batch_labels[:, :, -1]
    return last_columns  

def combine_train_data_and_labels(center_lat, center_lon, t_train, L, M, t_gap, t_span, lat_lon_window_size):

    global sst_data 

    try:
        t_datetime = datetime.strptime(str(t_train), "%Y%m%d")
    except ValueError:
        print(f"Invalid t_train format: {t_train}")
        return np.array([]), np.array([])

    train_data = []
    train_labels = []

    for span in range(t_span):

        current_t_train = t_datetime + timedelta(days=span)
        current_t_train_str = current_t_train.strftime("%Y%m%d")


        window_data = extract_region_temperature(
            center_lat, 
            center_lon, 
            lat_lon_window_size, 
            lat_lon_window_size, 
            current_t_train_str, 
            M, 
            t_gap
        )
        
        if window_data.size == 0:
            print(f"No window data for t_train: {current_t_train_str}")
            continue


        window_data = window_data.T  


        N = window_data.shape[0]

        if N == 0:
            print(f"No points in window data for t_train: {current_t_train_str}")
            continue

        middle_row_index = (N + 1) // 2 - 1 

        middle_row = window_data[middle_row_index]

        window_data_rest = np.delete(window_data, middle_row_index, axis=0)

        window_data = np.vstack([middle_row, window_data_rest])


        train_data.append(window_data)  

        center_point_temp = extract_center_point_temperature(
            center_lat, 
            center_lon, 
            current_t_train_str, 
            L + M, 
            t_gap
        )
        
        if center_point_temp.size == 0:
            print(f"No center point temp for t_train: {current_t_train_str}")
            train_labels.append(np.full((L, M), np.nan))
            continue

        delay_embedding_matrix = construct_delay_embedding_matrix(center_point_temp, L, M)
        
        if delay_embedding_matrix.size == 0:
            print(f"Delay embedding matrix creation failed for t_train: {current_t_train_str}")
            train_labels.append(np.full((L, M), np.nan))
            continue

        
        train_labels.append(delay_embedding_matrix) 

    train_data = np.array(train_data)  
    train_labels = np.array(train_labels)  
    
    train_data = clean_nan_data(train_data)

    return train_data, train_labels

def integration(params):

    global sst_data 

    data_directory = params['data_directory']
    center_lat = params['center_lat']
    center_lon = params['center_lon']
    lat_lon_window_size = params['lat_lon_window_size']
    sampling_window = params['sampling_window']
    t_train = params['t_train']
    L = params['L']
    M = params['M']
    t_gap = params['t_gap']
    t_span = params['t_span']
    test_span = params['test_span']
    


    try:
        t_train_date = datetime.strptime(str(t_train), "%Y%m%d")
    except ValueError:
        print(f"Invalid t_train format in integration: {t_train}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    t_test_date = t_train_date + timedelta(days=t_span * t_gap + L)
    t_test_str = int(t_test_date.strftime("%Y%m%d"))

    year_start = t_train_date.year
    year_end = t_test_date.year+1

    open_sst_dataset(
        data_directory=data_directory,
        start_year=year_start,
        end_year=year_end,
        center_lat=center_lat,
        center_lon=center_lon,
        lat_span=lat_lon_window_size,
        lon_span=lat_lon_window_size,
        sampling_window=sampling_window  
    )

    lat_resolution = 0.25
    lon_resolution = 0.25

    lat_shifts = np.arange(-sampling_window, sampling_window + lat_resolution, lat_resolution)
    lon_shifts = np.arange(-sampling_window, sampling_window + lon_resolution, lon_resolution)
    shift_combinations = [(lat_shift, lon_shift) for lat_shift in lat_shifts for lon_shift in lon_shifts]

    train_data_all = []
    train_labels_all = []
    test_data_all = []
    test_labels_all = []

    for lat_shift, lon_shift in shift_combinations:
        shifted_center_lat = center_lat + lat_shift
        shifted_center_lon = center_lon + lon_shift

        center_point_temp = extract_center_point_temperature(shifted_center_lat, shifted_center_lon, t_train, 1,t_gap)

        if np.isnan(center_point_temp).any():
            continue  

        train_data, train_labels = combine_train_data_and_labels(
            shifted_center_lat,
            shifted_center_lon,
            t_train,
            L,
            M,
            t_gap,
            t_span,
            lat_lon_window_size
        )

        
        t_train_date = datetime.strptime(str(params['t_train']), "%Y%m%d")

        t_test_date = t_train_date + timedelta(days=params['t_span'] * params['t_gap']+params['L'])
        t_test = int(t_test_date.strftime("%Y%m%d"))
    
        test_data, test_labels = combine_train_data_and_labels(
            shifted_center_lat,
            shifted_center_lon,
            t_test,  
            L,
            M,
            t_gap,
            test_span,
            lat_lon_window_size
        )

        test_labels = process_last_column(test_labels)

        train_data_all.append(train_data)      
        train_labels_all.append(train_labels)  

        test_data_all.append(test_data)        
        test_labels_all.append(test_labels)    
    train_data_all=np.array(train_data_all)
    train_labels_all=np.array(train_labels_all)
    test_data_all=np.array(test_data_all)
    test_labels_all=np.array(test_labels_all)

    return train_data_all, train_labels_all, test_data_all, test_labels_all
