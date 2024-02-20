import h5py
import hdf5plugin
import os
import numpy as np


class eigerdata:
    def __init__(self):
        self.header = None
        self.data = None
        
    class header_object:
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
        
        
    def load_data(self,master_fp):
        self.header = self.header_object()
        
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
        
        data_temp = np.zeros([header_temp['TotalFrame'],header_temp['YPixelsInDetector'],header_temp['XPixelsInDetector']]) # create tank for data
        for sn in range(data_temp.shape[0]):
            print('Loading frame: %d/%d'%(sn+1,header_temp['TotalFrame']),end= '\r')
            data_temp[sn] = read_frame(master_fp,sn+1) # sn start from 0 but sheet start 1
        print('\t\t\t\t\tDone.')
        self.data = data_temp


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