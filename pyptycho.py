import eiger
import os
import pandas as pd
import numpy as np


class pyptycho_object:
    def __init__(self):
        self.scan_file_path = None
        self.raw_data = None
        self.scan_data = None
        

    def load_data(self):
        self.scan_data = self.read_bluesky_data()


    def read_bluesky_data(self,profile = '25A'):
        bluesky_data_fp = self.scan_file_path
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
        
        if profile == '25A':
            motor_direction_z = 1
            motor_direction_x = -1

        output = blueskydata()
        output.motor_direction_z = motor_direction_z
        output.motor_direction_x = motor_direction_x
        output.data_ff = data_ff
        output.master_fn = master_fn
        output.master_fp = os.path.join(data_ff,master_fn)
        output.scan_data_fn = scan_data_fn
        output.scan_data_fp = scanfile_fp
        output.readback_x = readback_x * 1e-6 # in meter
        output.readback_z = readback_z * 1e-6 # in meter
        output.set_x = set_x * 1e-6 # in meter
        output.set_z = set_z * 1e-6 # in meter
        
        # The definition of the axis of object is noticed in the ptt file.
        output.expo_pos_x = output.readback_x * - output.motor_direction_x# exposure position
        output.expo_pos_y = output.readback_z * - output.motor_direction_z
        
        return output

## class for blueskydata
class blueskydata:
    def __init__(self):
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