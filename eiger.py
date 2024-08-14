import h5py
import hdf5plugin
import os
import numpy as np
import pandas as pd

class __eiger_header_object:
        def __init__(self):
            self.h5MasterFilePath       = None
            self.BitDepthImage          = None
            self.SaturationIntensity    = None
            self.XPixelsInDetector      = None
            self.YPixelsInDetector      = None
            self.CountTime              = None
            self.DetectorDistance       = None
            self.XPixelSize             = None
            self.YPixelSize             = None
            self.Wavelength             = None
            self.BeamCenterX            = None
            self.BeamCenterY            = None
            self.PixelMask              = None
            self.TotalFrame             = None
            self.Energy                 = None
            
class __bluesky_data_object:
        def __init__(self):
            self.profile = 'bluesky'
            self.beamline = None
            self.motor_direction_z = None
            self.motor_direction_x = None
            self.data_ff = None
            self.master_fn = None
            self.master_fp = None
            self.scan_data_fn = None
            self.scan_data_fp = None
            self.readback_x = None
            self.readback_z = None
            self.set_x = None
            self.set_z = None
            self.expo_pos_x = None
            self.expo_pos_y = None

class __bluesky_exp_object:
    def __init__(self):
        self.header = None
        self.data = None
        self.scandata = None
        
        
def load_bluesky_exp_data(bluesky_scandata_filepath):
    # load scan data information
    bluesky_object = __bluesky_exp_object()
    
    # read bluesky scan data
    bluesky_object.scandata = read_bluesky_data(bluesky_scandata_filepath)
    
    # load header
    master_fp = bluesky_object.scandata.master_fp
    bluesky_object.header = read_header(master_fp)

    # load data        
    data_temp = np.zeros([bluesky_object.header.TotalFrame,bluesky_object.header.YPixelsInDetector,bluesky_object.header.XPixelsInDetector]) # create tank for data
    for sn in range(data_temp.shape[0]):
        print('Loading frame: %d/%d'%(sn+1,bluesky_object.header.TotalFrame),end= '\r')
        data_temp[sn] = read_frame(master_fp,sn+1) # sn start from 0 but sheet start 1
    print('\t\t\t\t\tDone.')
    bluesky_object.data = data_temp
    
    return bluesky_object
    

def read_bluesky_data(bluesky_data_fp,beamline = 'TPS 25A'):
    # for bluesky scan file at TPS 25A and home-made scan file at TPS 23A
    if not os.path.isfile(bluesky_data_fp):
        print('Bluesky scan file does not exist.')
        return
    
    data_ff, scan_data_fn = os.path.split(bluesky_data_fp)
    
    table_buffer = pd.read_csv(bluesky_data_fp,engine='python',sep=',| ')

    ## check eiger type
    if len([col for col in table_buffer.columns if 'eig1m' in col]) != 0: # Eiger 1M case
        file_name_pattern = table_buffer['eig1m_file_file_write_name_pattern'][0]
    elif len([col for col in table_buffer.columns if 'eig16m' in col]) != 0: # Eiger 16M case
        file_name_pattern = table_buffer['eig16m_file_file_write_name_pattern'][0]
    else:
        print('No Eiger data found.')

    master_fn = file_name_pattern + '_master.h5'
    scanfile_fp = bluesky_data_fp
    readback_x = table_buffer['_cisamf_x'].to_numpy() # in um
    readback_z = table_buffer['_cisamf_z'].to_numpy() # in um
    set_x = table_buffer['_cisamf_x_user_setpoint'].to_numpy() # in um
    set_z = table_buffer['_cisamf_z_user_setpoint'].to_numpy() # in um
    
    if beamline == 'TPS 25A':
        motor_direction_z = 1
        motor_direction_x = -1

    object = __bluesky_data_object()
    object.beamline = beamline
    object.motor_direction_z = motor_direction_z
    object.motor_direction_x = motor_direction_x
    object.data_ff = data_ff
    object.master_fn = master_fn
    object.master_fp = os.path.join(data_ff,master_fn)
    object.scan_data_fn = scan_data_fn
    object.scan_data_fp = scanfile_fp
    object.readback_x = readback_x * 1e-6 # in meter
    object.readback_z = readback_z * 1e-6 # in meter
    object.set_x = set_x * 1e-6 # in meter
    object.set_z = set_z * 1e-6 # in meter

    return object



## basic eigr read function
def read_header(master_fp):
    header_object = __eiger_header_object() # create object
    
    if not os.path.isfile(master_fp):
        print('File does not exist.')
        return

    fid = h5py.File(master_fp,'r') # open file object
    
    # find max frame number
    dset = list(fid['entry/data/'].items())
    for dsn in reversed(range(len(dset))):
        if dset[dsn][1] != None:
            TotalFrame = dset[dsn][1].attrs['image_nr_high']
            break
    
    header_object.h5MasterFilePath     = master_fp
    header_object.BitDepthImage        = fid['/entry/instrument/detector/bit_depth_image'][()]
    header_object.SaturationIntensity  = fid['/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff'][()]
    header_object.XPixelsInDetector    = fid['/entry/instrument/detector/detectorSpecific/x_pixels_in_detector'][()]
    header_object.YPixelsInDetector    = fid['/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'][()]
    header_object.CountTime            = fid['/entry/instrument/detector/count_time'][()]
    header_object.DetectorDistance     = fid['/entry/instrument/detector/detector_distance'][()]
    header_object.XPixelSize           = fid['/entry/instrument/detector/x_pixel_size'][()]
    header_object.YPixelSize           = fid['/entry/instrument/detector/y_pixel_size'][()]
    header_object.Wavelength           = fid['/entry/instrument/beam/incident_wavelength'][()]*1E-10 # convert from A to meter
    header_object.BeamCenterX          = fid['/entry/instrument/detector/beam_center_x'][()]
    header_object.BeamCenterY          = fid['/entry/instrument/detector/beam_center_y'][()]
    header_object.PixelMask            = fid['/entry/instrument/detector/detectorSpecific/pixel_mask'][()].astype(bool) # convert the mask to logical array
    header_object.TotalFrame           = TotalFrame
    h=6.62607015*10**-34
    e=1.60217651019*10**-19
    c=299792458.0
    header_object.Energy               = (h*c)/(header_object.Wavelength*e)

    
    fid.close()
    return header_object

def read_frame(master_fp,req):
    # req is the frame number
    # list(fid['entry/data/data_000002'].attrs) can show the attrs field name
    # fid['entry/data/data_000002'].attrs['image_nr_high'] can show the value
    
    fid = h5py.File(master_fp,'r') # open file object
     
    if req <= 0 :
        print('Request frame index <= 0.')
        return
     
    dset = list(fid['entry/data/'].items())
    fid.close()
    
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