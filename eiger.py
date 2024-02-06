import h5py
import hdf5plugin
import os
import numpy as np
import pandas

def load_header(fp):
    if not os.path.isfile(fp):
        print('File does not exist.')
        return

    fid = h5py.File(fp,'r') # open file object
    
    header = dict()
    # field can be browsed by list(fid.keys()), list(fid['entry/data/'].keys()) etc.
    header['h5MasterFilePath'] = fp
    header['BitDepthImage'] = fid['/entry/instrument/detector/bit_depth_image'][()]
    header['SaturationIntensity'] = fid['/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff'][()]
    header['XPixelsInDetector'] = fid['/entry/instrument/detector/detectorSpecific/x_pixels_in_detector'][()]
    header['YPixelsInDetector'] = fid['/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'][()]
    header['CountTime'] = fid['/entry/instrument/detector/count_time'][()]
    header['DetectorDistance'] = fid['/entry/instrument/detector/detector_distance'][()]
    header['XPixelSize'] = fid['/entry/instrument/detector/x_pixel_size'][()]
    header['YPixelSize'] = fid['/entry/instrument/detector/y_pixel_size'][()]
    header['Wavelength'] = fid['/entry/instrument/beam/incident_wavelength'][()]*1E-10 # convert from A to meter
    header['BeamCenterX'] = fid['/entry/instrument/detector/beam_center_x'][()]
    header['BeamCenterY'] = fid['/entry/instrument/detector/beam_center_y'][()]
    PixelMask = fid['/entry/instrument/detector/detectorSpecific/pixel_mask'][()].astype(bool) # convert the mask to logical array
    header['PixelROI'] = np.invert(PixelMask)
    header['ManualROI'] = np.ones([header['YPixelsInDetector'],header['XPixelsInDetector']],dtype = bool)
    header['ROI'] = None
    
    fid.close()
    return header

def __load_single_frame(fid,req):
    # req is the frame number
    # list(fid['entry/data/data_000002'].attrs) can show the attrs field name
    # fid['entry/data/data_000002'].attrs['image_nr_high'] can show the value
     
    if req <= 0 :
        print('Request frame index <= 0.')
        return
     
    dset = list(fid['entry/data/'].items())

    for dsn in range(len(dset)):
        if dset[dsn][1] == None:
            print('Request frame index > maximum frame number.')
            break    
        
        image_nr_low = dset[dsn][1].attrs['image_nr_low']
        image_nr_high = dset[dsn][1].attrs['image_nr_high']
        sn_list = np.arange(image_nr_low,image_nr_high+1)
        target_ind = np.where(sn_list == req)[0]
        if target_ind.size == 0:
            continue
        else:
            target_ind = target_ind[0]
            data = dset[dsn][1][target_ind]
            return data
        
def load_frame(fp,req_list,norm_by_counttime = True):
    # req_list can be a int value, int list or int ndarray
    # req_list also can be a string 'a' or 'all' for load all frames in data
    # example:
    # load_frame('test.h5',3)
    # load_frame('test.h5',[1,3,5])
    # load_frame('test.h5','all')
    
    if not os.path.isfile(fp):
        print('File does not exist.')
        return
    else:
        fid = h5py.File(fp,'r') # open file object
    
    if isinstance(req_list,int): # load single
        req_list = np.array([req_list])
    elif isinstance(req_list,list): # load in a list
        req_list = np.array(req_list)
    elif isinstance(req_list,str): # load all
        dset = list(fid['entry/data/'].items())
        for dsn in range(len(dset)):
            if dset[dsn][1] == None:
                break
            else:
                num_total_frame = dset[dsn][1].attrs['image_nr_high']
        req_list = np.arange(1,num_total_frame+1)
          
    num_req = len(req_list)
    XPixelsInDetector = fid['/entry/instrument/detector/detectorSpecific/x_pixels_in_detector'][()]
    YPixelsInDetector = fid['/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'][()]
    data = np.zeros((num_req,YPixelsInDetector,XPixelsInDetector))
    
    for sn in range(num_req):
        data[sn] = __load_single_frame(fid,req_list[sn])
        
    if norm_by_counttime:
        counttime = fid['/entry/instrument/detector/count_time'][()]
        data = data / counttime
    
    return data
        
def load_bluesky_data(bluesky_data_fp,profile = '25A'):
    # for bluesky scan file at TPS 25A and home-made scan file at TPS 23A
    if not os.path.isfile(bluesky_data_fp):
        print('Bluesky scan file does not exist.')
        return
    
    data_ff, scan_data_fn = os.path.split(bluesky_data_fp)
    
    table_buffer = pandas.read_csv(bluesky_data_fp,sep=',')
    output = dict()
    file_name_pattern = table_buffer['eig1m_file_file_write_name_pattern'][0]
    master_fn = file_name_pattern + '_master.h5'
    scanfile_fp = bluesky_data_fp
    readback_x = table_buffer['_cisamf_x'].to_numpy() # in um
    readback_z = table_buffer['_cisamf_z'].to_numpy() # in um
    set_x = table_buffer['_cisamf_x_user_setpoint'].to_numpy() # in um
    set_z = table_buffer['_cisamf_z_user_setpoint'].to_numpy() # in um
    
    if profile == '25A':
        motor_direction_z = 1
        motor_direction_x = -1
    
    output['motor_direction_z'] = motor_direction_z
    output['motor_direction_x'] = motor_direction_x
    output['data_ff'] = data_ff
    output['master_fn'] = master_fn
    output['master_fp'] = os.path.join(data_ff,master_fn)
    output['scan_data_fn'] = scan_data_fn
    
    output['readback_x'] = readback_x * 1e-6 # in meter
    output['readback_z'] = readback_z * 1e-6 # in meter
    output['set_x'] = set_x * 1e-6 # in meter
    output['set_z'] = set_z * 1e-6 # in meter
    
    # The definition of the axis of object is noticed in the ptt file.
    output['expo_pos_x'] = output['readback_x'] * - motor_direction_x# exposure position
    output['expo_pos_y'] = output['readback_z'] * - motor_direction_z
    
    return output

def load_exp_cond(bluesky_data_fp):
    scan_data_info = load_bluesky_data(bluesky_data_fp)
    h5_info = load_header(scan_data_info['master_fp'])
    
    exp_cond = scan_data_info | h5_info
    
    return exp_cond