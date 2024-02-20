import h5py
import hdf5plugin
import os
import numpy as np
import pandas as pd


class bluesky_scanning_data:
    def __init__(self):
        self.header = None
        self.data = None
        self.scandata = None
        
    class eiger_header_object:
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
            self.PixelROI               = None
            self.ManualROI              = None
            self.ROI                    = None
            self.TotalFrame             = None

    class bluesky_data_object:
        def __init__(self):
            self.profile = 'bluesky'
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
        
    def load_data(self,bluesky_scandata_filepath):
        # load scan data information
        self.scandata = self.__load_bluesky_data(bluesky_scandata_filepath)
        
        # load header
        master_fp = self.scandata.master_fp
        self.header = self.eiger_header_object()
        header_temp = read_header(master_fp)
        self.header.h5MasterFilePath       = header_temp['h5MasterFilePath']     
        self.header.BitDepthImage          = header_temp['BitDepthImage']        
        self.header.SaturationIntensity    = header_temp['SaturationIntensity']  
        self.header.XPixelsInDetector      = header_temp['XPixelsInDetector']    
        self.header.YPixelsInDetector      = header_temp['YPixelsInDetector']    
        self.header.CountTime              = header_temp['CountTime']            
        self.header.DetectorDistance       = header_temp['DetectorDistance']     
        self.header.XPixelSize             = header_temp['XPixelSize']           
        self.header.YPixelSize             = header_temp['YPixelSize']           
        self.header.Wavelength             = header_temp['Wavelength']           
        self.header.BeamCenterX            = header_temp['BeamCenterX']          
        self.header.BeamCenterY            = header_temp['BeamCenterY']          
        self.header.PixelROI               = header_temp['PixelROI']             
        self.header.ManualROI              = header_temp['ManualROI']            
        self.header.ROI                    = header_temp['ROI']                  
        self.header.TotalFrame             = header_temp['TotalFrame']
        
        # load data        
        data_temp = np.zeros([header_temp['TotalFrame'],header_temp['YPixelsInDetector'],header_temp['XPixelsInDetector']]) # create tank for data
        for sn in range(data_temp.shape[0]):
            print('Loading frame: %d/%d'%(sn+1,header_temp['TotalFrame']),end= '\r')
            data_temp[sn] = read_frame(master_fp,sn+1) # sn start from 0 but sheet start 1
        print('\t\t\t\t\tDone.')
        self.data = data_temp
    
    def __load_bluesky_data(self,scan_file_path,beamline = 'TPS 25A'):
        bluesky_data_fp = scan_file_path
        # for bluesky scan file at TPS 25A and home-made scan file at TPS 23A
        if not os.path.isfile(bluesky_data_fp):
            print('Bluesky scan file does not exist.')
            return
        
        data_ff, scan_data_fn = os.path.split(bluesky_data_fp)
        
        table_buffer = pd.read_csv(bluesky_data_fp,engine='python',sep=',| ')
    
        file_name_pattern = table_buffer['eig1m_file_file_write_name_pattern'][0]
        master_fn = file_name_pattern + '_master.h5'
        scanfile_fp = bluesky_data_fp
        readback_x = table_buffer['_cisamf_x'].to_numpy() # in um
        readback_z = table_buffer['_cisamf_z'].to_numpy() # in um
        set_x = table_buffer['_cisamf_x_user_setpoint'].to_numpy() # in um
        set_z = table_buffer['_cisamf_z_user_setpoint'].to_numpy() # in um
        
        if beamline == 'TPS 25A':
            motor_direction_z = 1
            motor_direction_x = -1

        bluesky_data_object = self.bluesky_data_object()
        bluesky_data_object.motor_direction_z = motor_direction_z
        bluesky_data_object.motor_direction_x = motor_direction_x
        bluesky_data_object.data_ff = data_ff
        bluesky_data_object.master_fn = master_fn
        bluesky_data_object.master_fp = os.path.join(data_ff,master_fn)
        bluesky_data_object.scan_data_fn = scan_data_fn
        bluesky_data_object.scan_data_fp = scanfile_fp
        bluesky_data_object.readback_x = readback_x * 1e-6 # in meter
        bluesky_data_object.readback_z = readback_z * 1e-6 # in meter
        bluesky_data_object.set_x = set_x * 1e-6 # in meter
        bluesky_data_object.set_z = set_z * 1e-6 # in meter
        
        # The definition of the axis of object is noticed in the ptt file.
        bluesky_data_object.expo_pos_x = bluesky_data_object.readback_x * - bluesky_data_object.motor_direction_x# exposure position
        bluesky_data_object.expo_pos_y = bluesky_data_object.readback_z * - bluesky_data_object.motor_direction_z
        
        return bluesky_data_object


def read_header(master_fp):
    if not os.path.isfile(master_fp):
        print('File does not exist.')
        return

    fid = h5py.File(master_fp,'r') # open file object
    
    header = dict()
    # field can be browsed by list(fid.keys()), list(fid['entry/data/'].keys()) etc.

    # find max frame number
    dset = list(fid['entry/data/'].items())
    for dsn in range(len(dset)):
        if dset[dsn][1] == None:
            TotalFrame = dset[dsn-1][1].attrs['image_nr_high']
            break
    header['h5MasterFilePath']     = master_fp
    header['BitDepthImage']        = fid['/entry/instrument/detector/bit_depth_image'][()]
    header['SaturationIntensity']  = fid['/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff'][()]
    header['XPixelsInDetector']    = fid['/entry/instrument/detector/detectorSpecific/x_pixels_in_detector'][()]
    header['YPixelsInDetector']    = fid['/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'][()]
    header['CountTime']            = fid['/entry/instrument/detector/count_time'][()]
    header['DetectorDistance']     = fid['/entry/instrument/detector/detector_distance'][()]
    header['XPixelSize']           = fid['/entry/instrument/detector/x_pixel_size'][()]
    header['YPixelSize']           = fid['/entry/instrument/detector/y_pixel_size'][()]
    header['Wavelength']           = fid['/entry/instrument/beam/incident_wavelength'][()]*1E-10 # convert from A to meter
    header['BeamCenterX']          = fid['/entry/instrument/detector/beam_center_x'][()]
    header['BeamCenterY']          = fid['/entry/instrument/detector/beam_center_y'][()]
    header['PixelROI']             = np.invert(fid['/entry/instrument/detector/detectorSpecific/pixel_mask'][()].astype(bool)) # convert the mask to logical array
    header['ManualROI']            = np.ones([header['YPixelsInDetector'],header['XPixelsInDetector']],dtype = bool)
    header['ROI']                  = None
    header['TotalFrame']           = TotalFrame

    
    fid.close()
    return header

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