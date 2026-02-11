import numpy as np
from matplotlib.colors import ListedColormap

class Maps:
    
    def __init__(self, Data=None):
        
        if Data is None:
            pass
        else:
            self.Data = Data
            
    def Half_Amp_Latency(self, Trace, startPt=None, numPt=None):
        '''
        Calculate the Half Amplitude Latency of a given Trace (1-D). The final results are the moments after the start point of the time window.
        '''
        
        max_amplitude = np.max(Trace[startPt:startPt+numPt])
        half_amp = max_amplitude / 2.0
        value = 0
        
        for i in range(startPt, startPt+numPt):
            if Trace[i] == half_amp:
                value = i/2
                break
            elif Trace[i] > half_amp:
                value = (i - (Trace[i] - half_amp) / (Trace[i] - Trace[i-1])) / 2
                break
            
        return value
    
    def colorbar(self):
        colors = []

    # Black -> Blue
        for j in range(256):
            colors.append([0, 0, j])

    # Blue -> Cyan
        for j in range(1, 256):
            colors.append([0, j, 255])

        k = 0
    # Cyan -> Green
        for j in range(254, -1, -1):
            colors.append([0, 254 - k // 8, j])
            k += 1

    # Green -> Yellow
        for j in range(1, 256):
            colors.append([j, 255 - k // 8, 0])
            k -= 1

    # Yellow -> Red
        for j in range(254, -1, -1):
            colors.append([255, j, 0])

        lut = np.array(colors, dtype=np.float32) / 255.0
        
        cmap_cpp = ListedColormap(lut)
        
        return cmap_cpp
    
    def SNR_Map(self, startPt, numPt, Data=None):

        if Data is None:
            Data = self.Data
        
        Data_ave = np.mean(Data, axis=0)

        map = np.zeros((80, 80))

        for i in range(80):
            for j in range(80):
                SD = np.std(Data_ave[i, j, 10:60], ddof=1)
                MaxAmp = np.max(Data_ave[i, j, startPt:(startPt+numPt)])
                SNR = MaxAmp / SD
                map[i][j] = SNR
        
        map = map / np.max(map)

        return map