import sys
sys.path.append('/home/kevinteng/Desktop/BrainTumourSegmentation')
import tensorflow as tf
import numpy as np
import os



def threeD_to_twoD02(save_dir, input_path, n):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    patients = sorted(os.listdir(input_path))
    n_patients = len(patients)
    stacks = n_patients // n
    remainder = n_patients % n
    for j in range(stacks + 1):
        np_stack00 = 0  # flush ram
        np_stack00 = []
        # divisible stack
        if j < stacks:
            n_end = n * (j + 1)
        # remainder
        else:
            n_end = n * j + remainder
        # debugger
        print("i: {} with n_end:{}".format(j, n_end))

        for idx in range(n * j, n_end):
            merge_path_01 = os.path.join(input_path + patients[idx])
            med_img = np.load(merge_path_01).astype(np.float32)
            for i in range(med_img.shape[0]):  # first dimension is the number of slices
                if np.max(med_img[i, :, :, 0]) == np.min(med_img[i, :, :, 0]):  # remove empty slices
                    continue
                else:
                    np_stack00.append(med_img[i])
        np_stack00 = tf.convert_to_tensor(np_stack00)
        np.save(save_dir + 'stack_{}.npy'.format(j), np.array(np_stack00))


input_path = "/home/kevinteng/Desktop/ssd02/BraTS2020_preprocessed05/Training_pre/"
save_dir = "/home/kevinteng/Desktop/ssd02/BraTS2020_stack05/"

if __name__ == "__main__":
    threeD_to_twoD02(save_dir, input_path, n=30)