"""Values used by make_figures.py, copied from the paper's tables.

dev:  relative MASE on the development countries, recomputed from Tables 5 and 6
      (DHBO-v2-CNN from Table 7).
held: relative MASE on the held-out countries (Table S9).
"""
import numpy as np
models=['Naive','ARIMA','SARIMA','ETS (damped)','LSTM','GRU','ARIMA–LSTM hybrid','DHBO-v1','DHBO-v2']
daily={'Naive':[0.249,3.386,10.105,2.510],'ARIMA':[0.180,3.969,3.532,3.646],'SARIMA':[0.177,4.408,5.028,3.078],
'ETS (damped)':[0.182,4.052,3.374,3.317],'LSTM':[0.184,3.387,3.748,2.696],'GRU':[0.190,3.417,4.132,3.066],
'ARIMA–LSTM hybrid':[0.179,3.923,3.524,3.539],'DHBO-v1':[0.163,3.816,4.052,2.469],'DHBO-v2':[0.170,3.299,3.550,2.490]}
cum={'Naive':[0.206,1.727,4.342,2.841],'ARIMA':[0.055,1.315,0.613,1.313],'SARIMA':[0.055,1.217,0.664,0.497],
'ETS (damped)':[0.040,1.261,1.217,1.345],'LSTM':[0.747,1.855,6.396,0.891],'GRU':[0.722,1.864,2.161,1.291],
'ARIMA–LSTM hybrid':[0.111,1.944,0.418,0.892],'DHBO-v1':[0.159,0.797,0.487,0.663],'DHBO-v2':[0.043,0.829,0.450,0.482]}
def relgm(m):
    r=[a/b for a,b in zip(daily[m]+cum[m],daily['SARIMA']+cum['SARIMA'])]
    return float(np.exp(np.mean(np.log(r))))
dev={m:relgm(m) for m in models}; dev['DHBO-v2-CNN']=0.773
held={'DHBO-v1':0.754,'DHBO-v2':0.742,'DHBO-v2-CNN':0.762,'ARIMA–LSTM hybrid':0.745,'ETS (damped)':0.945,'LSTM':1.204,'GRU':1.213,'ARIMA':1.004,'SARIMA':1.000,'Naive':1.658}
if __name__=='__main__':
    for m in dev: print(f'{m:20s} dev {dev[m]:.3f}  held {held[m]:.3f}')
