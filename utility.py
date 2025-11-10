import numpy as np
import struct

class DataLoader:
    
    def __init__(self, filedir):
        '''
        Initialize the Data and Parameters.
        Preparation for further processing.
        '''
    
        # Important Index of the ZDA Data.
        self.data, metadata, self.rli, self.supplyment = self.from_zda_to_numpy(filedir)
        self.filedir = filedir
        self.meta = metadata
        
        # The four dimensions of the Data Array.
        self.trials = metadata['number_of_trials']
        self.points = metadata['points_per_trace'] - 24
        self.width = metadata['raw_width']
        self.height = metadata['raw_height']
        
           
    def from_zda_to_numpy(self, filedir):
        '''
        Read ZDA file and convert the data into numpy array which can be used in Python.
        Including RLI Data, MetaData, Raw Data, and Supplymental Data.
        '''
        
        # Load and read the binary ZDA file.
        file = open(filedir, 'rb')
        
        # Read in different scales.
        chSize = 1
        shSize = 2
        nSize = 4
        tSize = 8
        fSize = 4
        
        # MetaData.
        metadata = {}
        metadata['version'] = (file.read(chSize))
        metadata['slice_number'] = (file.read(shSize))
        metadata['location_number'] = (file.read(shSize))
        metadata['record_number'] = (file.read(shSize))
        metadata['camera_program'] = (file.read(nSize))

        metadata['number_of_trials'] = (file.read(chSize))
        metadata['interval_between_trials'] = (file.read(chSize))
        metadata['acquisition_gain'] = (file.read(shSize))
        metadata['points_per_trace'] = (file.read(nSize))
        metadata['time_RecControl'] = (file.read(tSize))

        metadata['reset_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['reset_duration'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['shutter_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['shutter_duration'] = struct.unpack('f',(file.read(fSize)))[0]

        metadata['stimulation1_onset'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['stimulation1_duration'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['stimulation2_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['stimulation2_duration'] = struct.unpack('f',(file.read(fSize)))[0]

        metadata['acquisition_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['interval_between_samples'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['raw_width'] = (file.read(nSize))
        metadata['raw_height'] = (file.read(nSize))

        for k in metadata:
            if k in ['interval_between_samples',] or 'onset' in k or 'duration' in k:
                pass # any additional float processing can go here
            elif k == 'time_RecControl':
                pass # TO DO: timestamp processing
            else:
                metadata[k] = int.from_bytes(metadata[k], "little") # endianness

        num_diodes = metadata['raw_width'] * metadata['raw_height']
        
        file.seek(1024, 0)
        
        # RLI 
        rli = {}
        rli['rli_low'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]
        rli['rli_high'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]
        rli['rli_max'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]
        
        # Raw Data.
        raw_data = np.zeros((metadata['number_of_trials'],
                             metadata['points_per_trace'],
                             metadata['raw_width'],
                             metadata['raw_height'])).astype(int)

        for i in range(metadata['number_of_trials']):
            for jw in range(metadata['raw_width']):
                for jh in range(metadata['raw_height']):
                    for k in range(metadata['points_per_trace']):
                        pt = file.read(shSize)
                        if not pt:
                            print("Ran out of points.",len(raw_data))
                            file.close()
                            return metadata
                        raw_data[i,k,jw,jh] = int.from_bytes(pt, "little")
        
        # Supplymental Data.
        supplyment = np.zeros((((metadata['number_of_trials']-1) * 8), metadata['points_per_trace']))
        
        for i in range(((metadata['number_of_trials']-1) * 8)):
            for j in range(metadata['points_per_trace']):
                pt = file.read(shSize)
                supplyment[i][j] = int.from_bytes(pt, "little")

        file.close()
        
        return raw_data, metadata, rli, supplyment
    
    def discard_and_rearrange(self):
        '''
        Discard and rearrange the useless (or noised) pixels and points.
        '''
        
        # Number of the Noised Data Points.
        number = 24
        
        # Discard the Noised Data Points.
        Data = np.copy(self.data[:, number:, :, :])
        Supplyment = np.copy(self.supplyment[:, number:])
        Data_rearrange = np.zeros((self.trials, self.height, self.width, self.points))
        
        # Rearrange the Data.
        for i in range(self.trials):
            for j in range(self.height):
                for k in range(self.width):
                    
                    Data_rearrange[i][j][k] = Data[i, :, j, k]
        
        return Data_rearrange, Supplyment
    
    def fix_and_supply(self):
        '''
        Fix and supply the Data.
        '''
        
        # Load the Rearranged Data.
        Data_Raw, Supplyment = self.discard_and_rearrange()
        
        # Fix the Data.
        for i in range(self.trials):
            if i == 0:
                pass
            elif i!=0 and i<(self.trials-1):
                Data = np.copy(Data_Raw[i])
                Data_rearrange = Data.reshape(self.height*self.width, self.points)
                Data_fix = np.delete(Data_rearrange, np.arange(0, (i*8), step=1), axis=0)
                
                Data_supply = np.copy(Data_Raw[i+1])
                Data_supply = Data_supply.reshape(self.height*self.width, self.points)
                Data_supply = Data_supply[:(i*8), :]
                
                Data_fix = np.concatenate((Data_fix, Data_supply), axis=0)
                Data_fix = Data_fix.reshape(self.height, self.width, self.points)
                
                Data_Raw[i] = Data_fix
            elif i == self.trials-1:
                Data = np.copy(Data_Raw[i])
                Data_rearrange = Data.reshape(self.height*self.width, self.points)
                Data_fix = np.delete(Data_rearrange, np.arange(0, (i*8), step=1), axis=0)
                
                Data_fix = np.concatenate((Data_fix, Supplyment), axis=0)
                Data_fix = Data_fix.reshape(self.height, self.width, self.points)
                
                Data_Raw[i] = Data_fix
                
        Data_Raw = -1 * Data_Raw
        
        return Data_Raw
    
    def clamp(self):
        '''
        Clamp the first point of each pixel at zero. 
        '''
        
        # Load the Data.
        Data = self.fix_and_supply()
        
        for i in range(self.trials):
            for j in range(self.height):
                for k in range(self.width):
                    
                    Data[i][j][k] = Data[i, j, k, :] - Data[i, j, k, 0]
                    
        return Data
    
    def get_data(self, rli_division=True):
        # TO DO: implement rli_division functionality
        if rli_division:
            print("rli_division functionality not yet implemented: utility.py:get_data method")
        return self.clamp()

    def get_index(self):
        
        return self.meta
    def get_rli(self):
        
        return self.rli