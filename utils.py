import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

def channel_standardization(image):
    '''
    Stanadrdization of image channel wise => Standard score
    Parameters:
        image: Input image
    Return:
        Standardized image, s.t. (pixel_value -)
    '''
    mean_val = np.mean(image, axis=-1)
    std_dev_val = np.std(image, axis=-1)
    output = (image - np.expand_dims(mean_val, axis=-1)) / (np.expand_dims(std_dev_val, axis=-1))
    # some val for std.dev = 0
    cast = np.nan_to_num(output)

    return cast


def concat_recursive(a, b, max_count, count=0):
    '''
    Recursively concatenate the image stacks with the next image stacks

    @param a: Top first image stacks
    @param b: Following image stacks
    '''
    while count < max_count - 1:
        c = np.concatenate((a, b), axis=0)
        a = c
        count += 1
        concat_recursive(a, b, max_count, count)
    return a
# ------------------------------------------------------functions for tf_records----------------------------


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image):
    '''
    Adding image and label info to TFRecords dataset
    '''
    feature = {
        'image': _bytes_feature(image)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecords(tfrecord_dir, image_paths):
    '''
    write TFRecords to appointed directory
    '''
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        for image in image_paths:
            img_bytes = tf.io.serialize_tensor(image)
            example = serialize_example(img_bytes)
            writer.write(example)


def read_tfrecord(serialized_example):
    '''
    read TFRecords from appointed directory
    '''
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=float)
    return image


def parse_tfrecord(tf_dir):
    tfrecord_dataset = tf.data.TFRecordDataset(tf_dir)
    parsed_dataset = tfrecord_dataset.map(read_tfrecord)
    return parsed_dataset
# --------------------------------------------------------------------------------------------------------


def std_norm(slice):
    """
    Removes 1% of the top and bottom intensities and perform
    normalization on the input 2D slice.
    """
    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    if np.std(slice) == 0:
        return slice
    else:
        slice = (slice - np.mean(slice)) / np.std(slice)
        return slice


def min_max_norm(images):
    """
    Min max normalization of images
    Parameters:
        images: Input stacked image list
    Return:
        Image list after min max normalization
    """
    m = np.max(images)
    mi = np.min(images)
    if m==mi:
        images = np.zeros_like(images)
    else:
        images = (images - mi) / (m - mi)
    return images


def normalize_modalities(Slice, mode=None):
    """
    Performs normalization on each modalities of input
    """
    assert mode != None, "Please in put [mode] type! 'std' for standard normalization, 'minmax' for minmax normalization"
    normalized_slices = np.zeros_like(Slice).astype(np.float32)
    for slice_ix in range(4):
        if mode == 'std':
            normalized_slices[..., slice_ix] = std_norm(Slice[..., slice_ix])
        if mode == 'minmax':
            normalized_slices[..., slice_ix] = min_max_norm(Slice[..., slice_ix])
    return normalized_slices


def dicesq(y_true, y_pred, smooth=1e-5):
    '''
    Modified dice coefficient as refer to: https://arxiv.org/abs/1606.04797
    :param y_true: Ground truth
    :param y_pred: Prediction from the model
    :return: Modified dice coefficient
    '''
    nmr = 2 * tf.reduce_sum(y_true * y_pred)
    dnmr = tf.reduce_sum(y_true ** 2) + tf.reduce_sum(y_pred ** 2) + smooth
    return (nmr / dnmr)


def dicesq_loss(y_true, y_pred):
    '''
    Modified dice coefficient loss
    :param y_true: Ground truth
    :param y_pred: Prediction from the model
    '''
    return 1 - dicesq(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1e-5):
    '''
    Dice coefficient for tensorflow
    :param y_true: Ground truth
    :param y_pred: Prediction from the model
    :return: dice coefficient
    '''
    # if input is not flatten
    if (tf.rank(y_true) != 1 and tf.rank(y_pred) != 1):
        y_true = tf.reshape(y_true, [-1])  # flatten
        y_pred = tf.reshape(y_pred, [-1])  # flatten
    # casting for label from int32 to float32 for computation
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    # if intersection==0:
    #     return 0.0
    # else:
    dc = (2.0 * intersection + smooth) / \
         (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return dc.numpy()


def dice_coef_loss(y_true, y_pred):
    '''
    Dice coefficient loss for IOU
    '''
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_bool(y_true, y_pred):
    '''
    Dice coefficient for tensorflow (boolean version)
    * None differiantiable!
    :param y_true: Ground truth
    :param y_pred: Prediction from the model
    :return: dice coefficient
    '''
    if (tf.rank(y_true) != 1 and tf.rank(y_pred) != 1):
        y_true = tf.reshape(y_true, [-1])  # flatten
        y_pred = tf.reshape(y_pred, [-1])  # flatten

    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)
    intersection = tf.math.count_nonzero(y_true & y_pred)

    size_i1 = tf.math.count_nonzero(y_true)
    size_i2 = tf.math.count_nonzero(y_pred)
    summation = size_i1 + size_i2

    if summation != 0:
        dc = (2.0 * tf.cast(intersection, tf.float32) / tf.cast(summation, tf.float32)).numpy()
    else:
        dc = 1.0

    return dc


def ss_metric(y_true, y_pred, label_type='binary', mode='global', smooth=1e-5):
    '''
    Compute sensitivity and specificity for groundtruth and prediction
    :param y_true: Ground truth
    :param y_pred: Prediction from the model
    :label_type: 'binary': input labels is binarized
                 'multi': mutli class labels
    :mode: 'local' compute the sensitivity label wise
           'global' compute the sensitivity overall
    :return: sensitivity & specificity
    '''
    # if input is not flatten
    if (tf.rank(y_true) != 1 and tf.rank(y_pred) != 1):
        y_true = tf.reshape(y_true, [-1])  # flatten
        y_pred = tf.reshape(y_pred, [-1])  # flatten
    # label types
    if label_type == 'binary':
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = (tp + smooth) / (tp + fn + smooth)
        specificity = (tn + smooth) / (tn + fp + smooth)
    if label_type == 'multi':
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        # true positive rate
        if mode == 'global':
            tp = np.trace(cm)
            tp_fn = np.sum(cm)
        else:  # local
            tp = np.diag(cm)
            tp_fn = np.sum(cm, 1)
        sensitivity = (tp + smooth) / (tp_fn + smooth)
        # true negative rate
        diag = np.diag(cm)
        tn = []
        for i in range(len(cm)):
            negs = np.sum([neg for neg in diag if neg != diag[i]])
            tn.append(negs)
        cm_copy = cm
        # make diagonal 0
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i == j:
                    cm_copy[i, j] = 0
        if mode == 'global':
            tn = np.sum(tn)
            fp = np.sum(cm_copy)
        else:  # local
            tn = np.array(tn)
            fp = np.sum(cm_copy, 0)
        specificity = (tn + smooth) / (tn + fp + smooth)
    return sensitivity, specificity


def compute_metric(y_true, y_pred, label_type='binary'):
    '''
    This function compute the metrics specify by BraTS competition
    which is dice coefficient, sensitivity, specificity
    :param y_true: Ground truth image
    :param y_pred: Prediction image from the model
    :param label_type: 'binary': input labels is binarized
             'multi': mutli class labels
    :return: dice coefficient, sensitivity & specificity list
            with order ['core', 'enhancing', 'complete']
    '''
    y_list = [y_true, y_pred]
    tumours = ['core', 'enhancing', 'complete']
    dc_output = []
    sens_output = []
    spec_output = []
    # compute dice coefficient for each tumour type
    for tumour_type in tumours:
        if label_type == 'multi':
            # label 1, 3(4)
            if tumour_type == 'core':
                y_true, y_pred = [np.where(((lbl == 1) | (lbl == 3)), lbl, 0) for lbl in y_list]
            # label 3(4)
            if tumour_type == 'enhancing':
                y_true, y_pred = [np.where(lbl == 3, lbl, 0) for lbl in y_list]
            # label 1,2,3,
            if tumour_type == 'complete':
                y_true, y_pred = [np.where(lbl > 0, lbl, 0) for lbl in y_list]
        if label_type == 'binary':
            # label 1, 3(4) =>1
            if tumour_type == 'core':
                y_true, y_pred = [np.where(((lbl == 1) | (lbl == 3)), 1, 0) for lbl in y_list]
            # label 3(4) =>1
            if tumour_type == 'enhancing':
                y_true, y_pred = [np.where(lbl == 3, 1, 0) for lbl in y_list]
            # label 1,2,3 =>1
            if tumour_type == 'complete':
                y_true, y_pred = [np.where(lbl > 0, 1, 0) for lbl in y_list]
        dc_list = []
        sens_list = []
        spec_list = []
        # only single images [240, 240]
        if y_true.ndim == 2:
            dc = dice_coef_bool(y_true, y_pred)
            sensitivity, specificity = ss_metric(y_true, y_pred)
            # append for each tumour type
            dc_output.append(dc)
            sens_output.append(sensitivity)
            spec_output.append(specificity)
            # batched images [?,240,240]
        else:
            for idx in range(len(y_true)):
                y_true_f = tf.reshape(y_true[idx], [-1])  # flatten
                y_pred_f = tf.reshape(y_pred[idx], [-1])  # flatten

                dc = dice_coef_bool(y_true_f, y_pred_f)
                sensitivity, specificity = ss_metric(y_true_f, y_pred_f)
                # store values
                dc_list.append(dc)
                sens_list.append(sensitivity)
                spec_list.append(specificity)
            # output [BATCH_SIZE, tumours_type]
            # taking the mean along the batch axis
            mean_ = lambda x: np.mean(x)
            dc_batch_mean = mean_(dc_list)
            sens_batch_mean = mean_(sens_list)
            spec_batch_mean = mean_(spec_list)
            # append for each tumour type
            dc_output.append(dc_batch_mean)
            sens_output.append(sens_batch_mean)
            spec_output.append(spec_batch_mean)
    # for each list the order is as following=> 'core','enhancing','complete'
    return dc_output, sens_output, spec_output


def compute_metric_dc(y_true, y_pred):
    """
    This function compute the dice coefficient metrics specify by BraTS competition
    :param y_true: Ground truth image
    :param y_pred: Prediction image from the model
    :return: dice coefficient with order ['core', 'enhancing', 'complete']
    """
    # one hot input images
    y_true = tf.keras.utils.to_categorical(y_true, num_classes=4)  # [240,240,4]
    y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=4)  # [240,240,4]
    y_list = [y_true, y_pred]
    tumours = ['core', 'enhancing', 'complete']
    dc_output = []
    # compute dice coefficient for each tumour type
    for tumour_type in tumours:
        # label 1,3(4)
        if tumour_type == 'core':
            channel = [1, 3]
        # label 3(4)
        if tumour_type == 'enhancing':
            channel = [3]
        # label 1,2,3,
        if tumour_type == 'complete':
            channel = [1, 2, 3]
        # list to store DC of different images in the batch
        dc_list = []
        # list to store DC of different channel
        dc_inner = []
        # only single images [240, 240, 4]
        if y_true.ndim == 2:
            for c in channel:
                dc = dice_coef_bool(y_true[..., c], y_pred[..., c])
                dc_inner.append(dc)
            # append for each tumour type
            dc_output.append(np.mean(dc_inner))
        # batched images [?,240,240]
        else:
            for idx in range(len(y_true)):  # batch size
                # computing DC for each channel
                for c in channel:
                    y_true_f = tf.reshape(y_true[idx, ..., c], [-1])  # flatten
                    y_pred_f = tf.reshape(y_pred[idx, ..., c], [-1])  # flatten

                    dc = dice_coef_bool(y_true_f, y_pred_f)
                    dc_inner.append(np.mean(dc))
                # store values for along the batch axis
                dc_list.append(dc_inner)

            # output [BATCH_SIZE, tumours_type]
            # taking the mean along the batch axis
            mean_ = lambda x: np.mean(x)
            dc_batch_mean = mean_(dc_list)
            # append for each tumour type
            dc_output.append(dc_batch_mean)

    # for each list the order is as following=> 'core','enhancing','complete'
    return dc_output

def data_aug(imgs):
    '''
    This function returns augmented images with 4 choices:
    choice 0 -> no augmentation applied
    choice 1 -> flip up and down
    choice 2 -> flip left and right
    choice 3 -> rotate n_rot*90 where n_rot is random integer
    @param imgs: image with size [b,w,h,c]
    @return: augmented images
    '''
    choice = np.random.randint(0,4)
    # no augmentation
    if choice==0:
        x = imgs
    # flip up and down
    if choice==1:
        x = tf.image.flip_up_down(imgs)
    # flip left and right
    if choice==2:
        x = tf.image.flip_left_right(imgs)
    # rotation based on angle
    if choice==3:
        n_rot = np.random.randint(1,4)
        x = tf.image.rot90(imgs, k=n_rot)
    return x
