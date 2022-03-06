from write_excel import *
from data import *
from generator_18 import *
from loss import *
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 Error
from keras.optimizers import Adam
import cv2
from metrics16 import mutil_accuracy, accuracy, mutil_hd, asd, hd
from my_model_18 import ResNetR3_attention_mutil_scale

learn_rate = 1e-4


def generator_test_patch(x):
    x = x[np.newaxis, ...]
    x1, x2, x3, x4 = GLCM(x, 160)
    train_patch, train_loc = vols_generator_patch(vol_name=x, num_data=1, patch_size=[64, 64, 64],
                                                  stride_patch=[32, 32, 32], out=2, num_images=100)
    train_patch1, a = vols_generator_patch(vol_name=x1, num_data=1, patch_size=[64, 64, 64],
                                           stride_patch=[32, 32, 32], out=2, num_images=100)
    train_patch2, b = vols_generator_patch(vol_name=x2, num_data=1, patch_size=[64, 64, 64],
                                           stride_patch=[32, 32, 32], out=2, num_images=100)
    train_patch3, c = vols_generator_patch(vol_name=x3, num_data=1, patch_size=[64, 64, 64],
                                           stride_patch=[32, 32, 32], out=2, num_images=100)
    train_patch4, d = vols_generator_patch(vol_name=x4, num_data=1, patch_size=[64, 64, 64],
                                           stride_patch=[32, 32, 32], out=2, num_images=100)
    return train_patch, train_loc, train_patch1, train_patch2, train_patch3, train_patch4


#
model = ResNetR3_attention_mutil_scale((64, 64, 64, 1), (64, 64, 64, 1), (64, 64, 64, 1), (64, 64, 64, 1),
                                       (64, 64, 64, 1))
weight_file = open('weight/weight_tu18/weight.txt')  # 训练数据的名字放到txt文件里
weight_strings = weight_file.readlines()
valid_file = open('train_data/txt_file/test.txt')
valid_strings = valid_file.readlines()
valid_mask_file = open('train_data/txt_file/test_mask_tumor3.txt')
valid_mask_strings = valid_mask_file.readlines()
for k in range(0, 20):
    st = weight_strings[k].strip()  # 文件名
    weight = 'weight/weight_tu18/' + st
    model.load_weights(weight, by_name=True)
    for j in range(len(valid_strings)):
        mask = np.empty((192, 192, 160, 4))
        mask2 = np.empty((192, 192, 160, 4))
        st = valid_strings[j].strip()  # 文件名
        sl = valid_mask_strings[j].strip()
        y_ones = np.ones((192, 192, 160))
        x_test, y_test, ones, affine1 = generator_data('train_data/test/', st, 'train_data/test_mask_tumor3/', sl)
        # x_test = x_test*y_ones
        test_vols, test_vols_loc, test_vols1, test_vols2, test_vols3, test_vols4 = generator_test_patch(x_test)
        pred = []
        for i in range(len(test_vols[0])):
            pred_temp = model.predict(
                [test_vols[0][i], test_vols1[0][i], test_vols2[0][i], test_vols3[0][i], test_vols4[0][i]])
            # pred_temp = model.predict(test_vols[j][i])   # len = 9
            pred_temp1 = pred_temp[5]
            mask[test_vols_loc[0][i][0].start:test_vols_loc[0][i][0].stop,
            test_vols_loc[0][i][1].start:test_vols_loc[0][i][1].stop,
            test_vols_loc[0][i][2].start:test_vols_loc[0][i][2].stop, :] += pred_temp1[0, :, :, :, :]
            mask2[test_vols_loc[0][i][0].start:test_vols_loc[0][i][0].stop,
            test_vols_loc[0][i][1].start:test_vols_loc[0][i][1].stop,
            test_vols_loc[0][i][2].start:test_vols_loc[0][i][2].stop, :] += np.ones(pred_temp1.shape[1:]).astype(
                'float32')

        pred_mask = mask / mask2

        pred_TC = pred_mask[:, :, :, 1]
        pred_WT = pred_mask[:, :, :, 2]
        pred_ET = pred_mask[:, :, :, 3]

        pred_TC[pred_TC > 0.001] = 1
        pred_TC[pred_TC < 0.001] = 0

        pred_WT[pred_WT > 0.2] = 1
        pred_WT[pred_WT < 0.2] = 0

        pred_ET[pred_ET > 0.001] = 1
        pred_ET[pred_ET < 0.001] = 0

        aa = np.zeros((192, 192, 160))
        bb = np.zeros((192, 192, 160))
        cc = np.zeros((192, 192, 160))

        aa[y_test == 1] = 1
        bb[y_test == 2] = 1
        cc[y_test == 4] = 1

        y_TC = aa + cc
        y_WT = aa + bb + cc
        y_ET = cc

        #   mask3 = np.argmax(pred_mask, axis=-1)
        #    x_test = np.reshape(x_test, (192, 192, 160))
        # nib.save(nib.Nifti1Image(x_test, affine1), 'predict/propose/test/' + str(j) + '_test.nii.gz')
        # nib.save(nib.Nifti1Image(y_test, affine1),'predict/propose/mask/'+str(j)+'_label.nii.gz')
        #  pred_image = best_map(y_test, mask3, [192, 192, 160])
        nib.save(nib.Nifti1Image(pred_TC + pred_WT + pred_ET, affine1),
                 'predict/pre_18/' + str(k) + '_' + str(j) + '_predict.nii.gz')

        #    pred.append(pred_mask)
        l1 = np.array([1])
        val1 = dice(pred_TC, y_TC, l1)
        val2 = dice(pred_WT, y_WT, l1)
        val3 = dice(pred_ET, y_ET, l1)
        # val_mean = np.mean(val)
        val_dice = np.mean([val1, val2, val3])
        val1 = str(val1)
        val2 = str(val2)
        val3 = str(val3)

        # val_mean = np.mean([val1,val2,val3])
        dice11 = [[val1, val2, val3, val_dice]]
        print(dice11)
        path_dice = 'predict/pre_18/dice.xls'
        write_excel_xls_append(path_dice, dice11)
"""
    #    acc, acc_bk = mutil_accuracy(y_test, pred_image)
        acc1 = accuracy(pred_TC, y_TC)
        acc2 = accuracy(pred_WT, y_WT)
        acc3 = accuracy(pred_ET, y_ET)
        val_acc=np.mean([acc1, acc2, acc3])
        acc1 = str(acc1)
        acc2 = str(acc2)
        acc3 = str(acc3)
        acc11 = [[acc1, acc2, acc3, val_acc]]
        print(acc11)
        path_acc = 'predict/pre_16/acc.xls'
        write_excel_xls_append(path_acc, acc11)

    #    write_excel_xls_append('predict/pre_16/acc.xls', acc)

     #   asd, asd_bk = mutil_asd(y_test, pred_image)
        asd1 = asd(pred_TC, y_TC)
        asd2 = asd(pred_WT, y_WT)
        asd3 = asd(pred_ET, y_ET)
        val_asd = np.mean([asd1, asd2, asd3])
        asd1 = str(asd1)
        asd2 = str(asd2)
        asd3 = str(asd3)
        asd11 = [[asd1, asd2, asd3, val_asd],]
        print(asd11)
        path_asd = 'predict/pre_16/asd.xls'
        write_excel_xls_append(path_asd, asd11)

   #     print(asd)
   #     write_excel_xls_append('predict/pre_16/asd.xls', asd)
        hd1 = hd(pred_TC, y_TC)
        hd2 = hd(pred_WT, y_WT)
        hd3 = hd(pred_ET, y_ET)
        val_hd = np.mean([hd1, hd2, hd3])
        hd1 = str(hd1)
        hd2 = str(hd2)
        hd3 = str(hd3)
        hd11 = [[hd1, hd2, hd3, val_hd],]
        print(hd11)
        path_hd = 'predict/pre_16/hd.xls'
        write_excel_xls_append(path_hd, hd11)
    #    hd, hd_bk = mutil_hd(y_test, pred_image)

    #    print(hd)
   #     write_excel_xls_append('predict/pre_16/hd.xls', hd)

"""



