import numpy as np
import math

class Tools:
    
    def __init__(self, Data=None):
        '''
        Load the Data that are in a shape of Trials * height * width * points.
        '''
        if Data is None:
            pass
        else:
            self.Data = Data
        
    def Polynomial(self, startPt=None, numPt=None, Data=None):
        '''
        Implement 3-degrees polynomial regression to the Original Data. The skip window can be set by adjust numPt and startPt.
        '''
        
        if Data is None:
            Data = self.Data
        
        index = np.linspace(0, (Data.shape[3]-1), Data.shape[3])
        
        if startPt is not None:
            if numPt is not None:
                index_d = np.delete(index, np.arange(startPt, startPt+numPt, step=1), axis=0)
                index_skip = True
        else:
            index_d = index
            index_skip = False
          
        for i in range(Data.shape[0]):
            for j in range(Data.shape[1]):
                for k in range(Data.shape[2]):
                    
                    if index_skip:
                        array = np.delete(np.copy(Data[i][j][k]), np.arange(startPt, startPt+numPt, step=1), axis=0)
                    else:
                        array = np.copy(Data[i][j][k])
                    
                    coeffs = np.polyfit(index_d, array, 3)
                    Data_fit = np.polyval(coeffs, index)
                    Data[i][j][k] = Data[i][j][k] - Data_fit
                    
        return Data
    
    def PolyGaussian(self, startPt=None, numPt=None, Data=None):
        '''
        Reproduce PhotoZ's algorithm.
        '''
        
        degree = 3
        
        for i in range(Data.shape[0]):
            for j in range(Data.shape[1]):
                for k in range(Data.shape[2]):
                    
                    data = Data[i][j][k]
                    X = [[0.0]*(degree+1) for _ in range(degree+1)]
                    Y = [0.0]*(degree+1)
                    A = [0.0]*(degree+1)

                    cx = [0.0]*7
                    cy = [0.0]*4

                    endPt = startPt + numPt
                    length = len(data)

                    for s in range(3, length):
                        if startPt <= s < endPt:
                            continue

                        x = [0.0]*7
                        x[0] = 1.0
                        y = data[s]

                        for t in range(1, 7):
                            x[t] = x[t-1] * s

                        for t in range(7):
                            cx[t] += x[t]

                        for t in range(4):
                            cy[t] += y * x[t]

                    for s in range(degree+1):
                        for t in range(degree+1):
                            X[s][t] = cx[s + (3 - t)]
                        Y[s] = cy[s]
                        
                    for index in range(4):
                        maxV = 0.0
                        maxI = index

                        for s in range(index, degree+1):
                            tmp = abs(X[s][index])
                            if tmp > maxV:
                                maxV = tmp
                                maxI = s

                        if maxI != index:
                            for s in range(index, degree+1):
                                X[index][s], X[maxI][s] = X[maxI][s], X[index][s]

                            Y[index], Y[maxI] = Y[maxI], Y[index]
                            
                        for s in range(index+1, degree+1):
                            m = X[s][index] / X[index][index]

                            for t in range(index+1, degree+1):
                                X[s][t] -= X[index][t] * m

                            Y[s] -= Y[index] * m

                    A[3] = Y[3] / X[3][3]
                    A[2] = (Y[2] - X[2][3]*A[3]) / X[2][2]
                    A[1] = (Y[1] - X[1][3]*A[3] - X[1][2]*A[2]) / X[1][1]
                    A[0] = (Y[0] - X[0][3]*A[3] - X[0][2]*A[2] - X[0][1]*A[1]) / X[0][0]

                    x_all = np.arange(length)
                    value = ((A[0]*x_all + A[1])*x_all + A[2])*x_all + A[3]
                    Data[i][j][k][s] -= value
                    
                    # print('Trial #{}, Row #{}, Column #{}.'.format(i, j, k))
        return Data
    
    def Polynormal(self, startPt=None, numPt=None, Data=None):
        '''
        A new Trying.
        '''
        
        # Parameters setting.
        length = Data.shape[-1]
        endPt = startPt + numPt
        s = np.arange(length)
        mask = (s >= 3) & ~((s >= startPt) & (s < endPt))
        xs = s[mask]
        xs = np.array(xs)
        V = np.column_stack((xs**3, xs**2, xs, np.ones_like(xs)))
        x_all = np.arange(length)
        
        for i in range(Data.shape[0]):
            for j in range(Data.shape[1]):
                for k in range(Data.shape[2]):
                    
                    data = np.asarray(Data[i][j][k], dtype=float)
                    ys = data[mask]
                    ys = np.array(ys)

                    A, *_ = np.linalg.lstsq(V, ys, rcond=None)

                    baseline = ((A[0]*x_all + A[1])*x_all + A[2])*x_all + A[3]

                    Data[i][j][k] -= baseline
                    print('Trial #{}, Row #{}, Column #{}.'.format(i, j, k))
        return Data
    
    def Rli_Division(self, Rli, Data=None):
        '''
        ΔF/F.
        '''
        
        # Load the RLI Data.
        rli_low = Rli['rli_low']
        rli_low = np.array(rli_low)
        rli_low = rli_low.reshape(80, 80)
        rli_high = Rli['rli_high']
        rli_high = np.array(rli_high)
        rli_high = rli_high.reshape(80, 80)
        rli_max = Rli['rli_max']
        rli_max = np.array(rli_max)
        rli_max = rli_max.reshape(80, 80)
        
        if Data is None:
            Data = self.Data
        
        for i in range(Data.shape[0]):
            for j in range(Data.shape[1]):
                for k in range(Data.shape[2]):
                    rli = (rli_high[j][k] - rli_low[j][k]) / 3276.8
                    rli = round(rli, 6)
                    Data[i][j][k] = Data[i][j][k] / (rli * 2340)
        
        return Data
    
    def T_filter(self, Data=None):
        '''
        Apply bionomial8 Temporal Filter to the Target Data.
        '''
        
        if Data is None:
            Data = self.Data
        
        for i in range(Data.shape[0]):
                    
            input = np.copy(Data[i])
        
            num = Data.shape[3]
            output = np.zeros((Data.shape[1], Data.shape[2], Data.shape[3]))

            output[:, :, 0] = (input[:, :, 4] + 8 * input[:, :, 3] + 28 * input[:, :, 2] + 56 * input[:, :, 1]) / 93
            output[:, :, 1] = (input[:, :, 5] + 8 * input[:, :, 4] + 28 * input[:, :, 3] + 56 * input[:, :, 2] 
                               + 70 * input[:, :, 1]) / 163
            output[:, :, 2] = (input[:, :, 6] + 8 * input[:, :, 5] + 28 * input[:, :, 4] 
                               + 56 * (input[:, :, 1] + input[:, :, 3]) + 70 * input[:, :, 2]) / 219
            output[:, :, 3] = (input[:, :, 7] + 8 * input[:, :, 6] + 28 * (input[:, :, 1] + input[:, :, 5]) 
                               + 56 * (input[:, :, 2] + input[:, :, 4]) + 70 * input[:, :, 3]) / 247

            end = num-4	
            for s in range(4, end):
                output[:, :, s] = (input[:, :, s-4]+input[:, :, s+4] + 8*(input[:, :, s-3]+input[:, :, s+3]) 
                                    + 28*(input[:, :, s-2]+input[:, :, s+2])+56*(input[:, :, s-1]+input[:, :, s+1])+70*input[:, :, s])/256

            output[:, :, num-4] = (input[:, :, num-8] + 8 * (input[:, :, num-7] + input[:, :, num-1]) 
                                    + 28 * (input[:, :, num-6] + input[:, :, num-2]) + 56 * (input[:, :, num-5] + input[:, :, num-3]) 
                                    + 70 * input[:, :, num-4]) / 255
            output[:, :, num-3] = (input[:, :, num-7] + 8 * input[:, :, num-6] + 28 * (input[:, :, num-5] + input[:, :, num-1]) 
                                    + 56 * (input[:, :, num-4] + input[:, :, num-2]) + 70 * input[:, :, num-3]) / 247
            output[:, :, num-2] = (input[:, :, num-6] + 8 * input[:, :, num-5] + 28 * input[:, :, num-4] 
                                    + 56 * (input[:, :, num-3] + input[:, :, num-1]) + 70 * input[:, :, num-2]) / 219
            output[:, :, num-1] = (input[:, :, num-5] + 8 * input[:, :, num-4] + 28 * input[:, :, num-3] 
                                    + 56 * input[:, :, num-2] + 70 * input[:, :, num-1]) / 163
            
            '''
            output[0] = (output[6]+output[10])/2
            output[1] = (output[7] + output[9]) / 2
            output[2] = (output[8] + output[10]) / 2
            output[3] = (output[9] + output[6]) / 2
            output[4] = (output[10] + output[7]) / 2
            output[5] = (output[11] + output[13]) / 2
            '''
            
            Data[i] = output
                
        return Data
    
    def S_filter(self, sigma, Data=None):
        '''
        Apply Gaussian Spatial Filter to the Target Data. Sigma here means the spatial constant which is used to calculate the ceterWeight.
        '''
        
        if Data is None:
            Data = self.Data
        
        Trials, height, width, points = Data.shape    
        proData = np.zeros((Trials, height*width, points))
        for i in range(Data.shape[0]):
            proData[i] = Data[i].reshape(height*width, points)
            
        centerWeight = np.exp((0.5)/(sigma**2))
        
        num_diodes, numPts = proData.shape[1], proData.shape[2]
        output = np.zeros_like(proData)

        for i in range(Data.shape[0]):
            for center_diode in range(num_diodes):
                
                x = center_diode % width
                y = center_diode // width

                acc = np.zeros(numPts)
                weight_sum = 0.0

                acc += proData[i][center_diode] * centerWeight
                weight_sum += centerWeight

                for j in range(9):
                    if j == 4:
                        continue

                    xoffset = (j % 3) - 1
                    yoffset = (j // 3) - 1
                    nx, ny = x + xoffset, y + yoffset

                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue

                    neighbor = nx + ny * width
            

                    acc += proData[i][neighbor]
                    weight_sum += 1.0

                if weight_sum > 0:
                    output[i][center_diode] = acc / weight_sum
                else:
                    output[i][center_diode] = 0.0
        
        output = output.reshape(Data.shape[0], Data.shape[1], Data.shape[2], Data.shape[3])
        
        return output