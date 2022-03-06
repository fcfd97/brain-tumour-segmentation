# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 22:23:16 2022

@author: yangyue
"""

import glob
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
import os
import nibabel as nib
import unet3d
from unet3d.model import isensee2017
from nilearn.image import reorder_img, new_img_like
from unet3d.utils.sitk_utils import resample_to_spacing, calculate_origin_offset
from metrices_1 import dice

base_data_dir = '../TestingData/*'    #base_size:字符串、数据目录,

def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)

def test( gpu_id ):
    
    #GPU配置
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    set_session(sess)
    
    #测试图像列表
    test_names =  glob.glob(base_data_dir)
    sum = 0
    
    for i in range(len(test_names)):
        
        test_name = test_names[i]
        test_name = glob.glob(test_name + '/*')
    #    print(i)
   #     print(test_name)
   #     print('\n')
        
        x1 = nib.load(test_name[0])
        affine = x1.affine.copy()
        x2 = nib.load(test_name[1])
        x3 = nib.load(test_name[2])
        x4 = nib.load(test_name[3])
    
        image_shape = [128,128,128]
     
        x1 = resize(x1, new_shape=image_shape, interpolation='linear')
        x2 = resize(x2, new_shape=image_shape, interpolation='linear')
        x3 = resize(x3, new_shape=image_shape, interpolation='linear')
        x4 = resize(x4, new_shape=image_shape, interpolation='linear')
    
        
        x1 = x1.get_fdata()
        x2 = x2.get_fdata()
        x3 = x3.get_fdata()
        x4 = x4.get_fdata()
    
        
        x1 = np.reshape(x1,(1,) + x1.shape)
        x2 = np.reshape(x2,(1,) + x2.shape)
        x3 = np.reshape(x3,(1,) + x3.shape)
        x4 = np.reshape(x4,(1,) + x4.shape)
    
        X = np.concatenate((x2,x3,x4,x1))
        X = np.reshape(X,(1,) + X.shape)
        
        model_name = glob.glob('../moxing/*')
        model = isensee2017.isensee2017_model(input_shape=(4,128,128,128),
                                  n_labels=3,
                                  initial_learning_rate=1e-4,
                                  n_base_filters=16)

        model.load_weights(model_name[0])
        pred = model.predict([X]) 
        y = np.rollaxis(pred[0,:,:,:,],0,4)
        
        y_1 = y[:,:,:,0]
        y_2 = y[:,:,:,1]
        y_3 = y[:,:,:,2]
        
        y_1[y_1 >= 0.9] = 1     #标签 1
        y_1[y_1 < 0.9] = 0
        
        y_2[y_2 >= 0.9] = 1
        y_2[y_2 < 0.9] = 0
        
        y_3[y_3 >= 0.9] = 1
        y_3[y_3 < 0.9] = 0
        
        y_22 = y_2 * 2
        y_33 = y_3 * 4
        
        output = y_1 + y_2 + y_3
        output = nib.Nifti1Image(output,affine)
        
        pred_output =   y_1 + y_22 + y_33
        pred_output = nib.Nifti1Image(pred_output,affine)
        
        pred = resize(pred_output, new_shape=[240,240,155], interpolation='nearest')
       # nib.save(pred, 'pred'+str(i)+'.nii')
       
       # da = resize(output, new_shape=[240,240,155], interpolation='nearest')
        data = pred.get_fdata()
        data = data.astype(np.int16)
        nib.save(nib.Nifti1Image(data,affine), 'predict/pre3/'+'pred'+str(i)+'.nii')
        
        
        
        
        
        jia3 = y_1 + y_2 + y_3
        jia2 = y_1 + y_3
        jia1 = y_3
        
        jia3 = resize(nib.Nifti1Image(jia3,affine), new_shape=[240,240,155], interpolation='nearest')
        jia3 = jia3.get_fdata()
        jia3 = jia3.astype(np.int16)
        
        
        jia2 = resize(nib.Nifti1Image(jia2,affine), new_shape=[240,240,155], interpolation='nearest')
        jia2 = jia2.get_fdata()
        jia2 = jia2.astype(np.int16)
        
        
        jia1 = resize(nib.Nifti1Image(jia1,affine), new_shape=[240,240,155], interpolation='nearest')
        jia1 = jia1.get_fdata()
        jia1 = jia1.astype(np.int16)
        
        
        label_names = glob.glob('../test_label/*')
        seg = label_names[i]
        seg = nib.load(seg)
        seg = seg.get_fdata()
        
        a = np.zeros((240,240,155))
        b = np.zeros((240,240,155))
        c = np.zeros((240,240,155))

        a[seg == 1] = 1
        b[seg == 2] = 1
        c[seg == 4] = 1
        
        
        seg3 = a + b + c
        seg2 = a + c
        seg1 = c
        
        
        
        labels = [1]
        
      #  print(label_names[i])

        val1, _ = dice(jia1,seg1,labels, nargout=2)
        val2, _ = dice(jia2,seg2,labels, nargout=2)
        val3, _ = dice(jia3,seg3,labels, nargout=2)
      #  vals,_ = dice(seg,data,[1,2,4], nargout=2)
       # print(vals)
        sum = sum + val1 + val2 + val3
        
        print(val1,val2,val3)
    print(sum/114)
        
       # print(i)
       
        
                    
    
if __name__ == "__main__":        
    test(1)


