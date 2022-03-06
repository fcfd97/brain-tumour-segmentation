import numpy as np
from binary import *


def mutil_seg(label):
    label = np.reshape(label,(192,192,160))
    BK = np.zeros((192,192,160))
    TM = np.zeros((192,192,160))
    CSF = np.zeros((192,192,160))
    GM = np.zeros((192,192,160))
    WM = np.zeros((192,192,160))
    BK[label ==0] =1
    TM[label==1] =1
    CSF[label==2] =1
    GM[label ==3] =1
    WM[label ==4] =1
    return BK,TM,CSF,GM,WM


def accuracy(y_true, y_pred):
    # https://stackoverflow.com/a/27475514
    y_true = np.reshape(y_true,(240*240*155,1))
    y_pred = np.reshape(y_pred,(240*240*155,1))
    true_positive = len(np.where((y_true == 1) & (y_pred == 1))[0])
    true_negative = len(np.where((y_true == 0) & (y_pred == 0))[0])
    false_positive = len(np.where((y_true == 0) & (y_pred == 1))[0])
    false_negative = len(np.where((y_true == 1) & (y_pred == 0))[0])
    acc = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    return acc
def mutil_accuracy(y_true, y_pred):
    T_BK,T_TM,T_CSF,T_GM,T_WM = mutil_seg(y_true)
    P_BK,P_TM,P_CSF,P_GM,P_WM = mutil_seg(y_pred)
    a1 = accuracy(T_TM,P_TM)
    a2 = accuracy(T_CSF,P_CSF)
    a3 = accuracy(T_GM,P_GM)
    a4 = accuracy(T_WM,P_WM)
    mean = (a1+a2+a3+a4)/4
    accurcy = [[a1,a2,a3,a4, mean], ]
    BK = accuracy(T_BK,P_BK)
    return accurcy,BK

def mutil_asd(y_true, y_pred):
    T_BK,T_TM,T_CSF,T_GM,T_WM = mutil_seg(y_true)
    P_BK,P_TM,P_CSF,P_GM,P_WM = mutil_seg(y_pred)
    a1 = asd(T_TM,P_TM)
    a2 = asd(T_CSF,P_CSF)
    a3 = asd(T_GM,P_GM)
    a4 = asd(T_WM,P_WM)
    mean = (a1+a2+a3+a4)/4
    asd1 = [[a1,a2,a3,a4, mean], ]
    BK = accuracy(T_BK,P_BK)
    return asd1,BK

def mutil_hd(y_true, y_pred):
    T_BK,T_TM,T_CSF,T_GM,T_WM = mutil_seg(y_true)
    P_BK,P_TM,P_CSF,P_GM,P_WM = mutil_seg(y_pred)
    a1 = hd(T_TM,P_TM)
    a2 = hd(T_CSF,P_CSF)
    a3 = hd(T_GM,P_GM)
    a4 = hd(T_WM,P_WM)
    mean = (a1+a2+a3+a4)/4
    hd1 = [[a1,a2,a3,a4, mean], ]
    BK = accuracy(T_BK,P_BK)
    return hd1,BK