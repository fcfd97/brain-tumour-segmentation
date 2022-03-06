import os
import glob
import numpy as np
import nibabel as nib
from metrics import mutil_accuracy, accuracy, mutil_hd, asd, hd
from write_excel import *
base_data_dir = '../data/brats20/'    #数据目录

mask_data = glob.glob(base_data_dir + 'mask18/*') #标准
seg_data = glob.glob(base_data_dir + 'output/*') #生成分割图


def dice1(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


if __name__ == "__main__":

    for k in range(10):


        y_test = nib.load(mask_data[k])
        pred_mask = nib.load(seg_data[k])

        y_test = y_test.get_fdata()
        pred_mask = pred_mask.get_fdata()

        aa = np.zeros((240, 240, 155))
        bb = np.zeros((240, 240, 155))
        cc = np.zeros((240, 240, 155))

        aa[y_test == 1] = 1
        bb[y_test == 2] = 1
        cc[y_test == 4] = 1

        y_TC = aa + cc
        y_WT = aa + bb + cc
        y_ET = cc


        aaa = np.zeros((240, 240, 155))
        bbb = np.zeros((240, 240, 155))
        ccc = np.zeros((240, 240, 155))

        aaa[pred_mask == 1] = 1
        bbb[pred_mask == 2] = 1
        ccc[pred_mask == 4] = 1

        pred_TC = aaa + ccc
        pred_WT = aaa + bbb + ccc
        pred_ET = ccc


    #    pred.append(pred_mask)
        l1 = np.array([1])

        val1 = dice1(pred_TC, y_TC, l1)
        a = val1[0]
        val2 = dice1(pred_WT, y_WT, l1)
        b = val2[0]
        val3 = dice1(pred_ET, y_ET, l1)
        c = val3[0]
        val_dice = np.mean([a, b ,c])
        print(val1,val2,val3)
        path_dice = 'E:\seg\BraTS-2020-master\BraTS-2020-master\output\dice.xls'
        dice11 = [[a, b, c,val_dice]]
        write_excel_xls_append(path_dice, dice11)
 #      acc, acc_bk = mutil_accuracy(y_test, pred_image)
"""
        acc1 = accuracy(pred_TC, y_TC)
        a = acc1[0]
        acc2 = accuracy(pred_WT, y_WT)
        b = acc2[0]
        acc3 = accuracy(pred_ET, y_ET)
        c = acc3[0]
        val_acc=np.mean([a, b, c])
        acc11 = [[a, b, c, val_acc]]
        print(acc11)
        path_acc = r'E:\seg\BraTS-2020-master\BraTS-2020-master\output\acc.xls'
        write_excel_xls_append(path_acc, acc11)
     #   asd, asd_bk = mutil_asd(y_test, pred_image)
        asd1 = asd(pred_TC, y_TC)
        a = asd1[0]
        asd2 = asd(pred_WT, y_WT)
        b = asd2[0]
        asd3 = asd(pred_ET, y_ET)
        c = asd3[0]
        val_asd = np.mean([a, b, c])
        asd11 = [[a, b, c, val_asd],]
        print(asd11)
        path_asd = r'E:\seg\BraTS-2020-master\BraTS-2020-master\output\asd.xls'
        write_excel_xls_append(path_asd, asd11)

        hd1 = hd(pred_TC, y_TC)
        a = hd1[0]
        hd2 = hd(pred_WT, y_WT)
        b = hd2[0]
        hd3 = hd(pred_ET, y_ET)
        c = hd3[0]
        val_hd = np.mean([a, b, c])
        hd11 = [[a, b, c, val_hd],]
        print(hd11)
        path_hd = 'E:\seg\BraTS-2020-master\BraTS-2020-master\output\hd.xls'
        write_excel_xls_append(path_hd, hd11)
"""
