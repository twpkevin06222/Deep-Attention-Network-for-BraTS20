import tensorflow as tf
from tensorflow.keras.layers import GaussianNoise, concatenate
from utils_model import *
from utils import *
from attention import *
from coord_conv import CoordConv

# template for guided attention block
layer_name_p01 = ['pam01_conv01', 'pam01_conv02', 'pam01_softmax', 'pam01_conv03',
                  'pam01_alpha','pam01_add']
layer_name_c01 = ['cam01_softmax', 'cam01_alpha','cam01_add']
layer_name_p02 = ['pam02_conv01', 'pam02_conv02', 'pam02_softmax', 'pam02_conv03',
                  'pam02_alpha', 'pam02_add']
layer_name_c02 = ['cam02_softmax', 'cam02_alpha','cam02_add']
layer_name_template = [layer_name_p01, layer_name_c01, layer_name_p02, layer_name_c02]

layer_name_ga = []
for b in range(1,4):
    layer_block = []
    for layer in layer_name_template:
        layer_internal = [i+'block0{}'.format(b) for i in layer]
        layer_block.append(layer_internal)
    layer_name_ga.append(layer_block)

hn = 'he_normal' #kernel initializer

def Unet_model(input_layer, dropout=0.2):
    # downsampling
    #     conv1 = coordconv_block(input_layer, x_dim=240, y_dim=240, filters=64)
    conv1 = conv_block(input_layer, filters=64, kernel_initializer=hn)
    pool1 = pool(conv1)

    conv2 = conv_block(pool1, filters=128, kernel_initializer=hn)
    pool2 = pool(conv2)

    conv3 = conv_block(pool2, filters=256, kernel_initializer=hn)
    pool3 = pool(conv3)

    conv4 = conv_block(pool3, filters=512, kernel_initializer=hn, dropout_rate=dropout)
    pool4 = pool(conv4)

    conv5 = conv_block(pool4, filters=1024, kernel_initializer=hn, dropout_rate=dropout)

    # upsampling
    up1 = up(conv5, filters=512, merge=conv4, kernel_initializer=hn)
    #     conv6 = coordconv_block(up1, x_dim=30, y_dim=30, filters=512)
    conv6 = conv_block(up1, filters=512, kernel_initializer=hn)

    up2 = up(conv6, filters=256, merge=conv3, kernel_initializer=hn)
    conv7 = conv_block(up2, filters=256, kernel_initializer=hn)

    up3 = up(conv7, filters=128, merge=conv2, kernel_initializer=hn)
    conv8 = conv_block(up3, filters=128, kernel_initializer=hn)

    up4 = up(conv8, filters=64, merge=conv1, kernel_initializer=hn)
    conv9 = conv_block(up4, filters=64, kernel_initializer=hn)

    output_layer = Conv2D(4, (1, 1), activation='softmax')(conv9)

    return output_layer


def AttUnet_model(input_layer, attention_mode='grid', dropout=0.2):
    '''
    Attention Unet without deep supervision
    @param input_layer: input batched image [b,w,h,c]
    @param attention_mode: choice of attention mode, default 'grid'
            where the gated signal derived from upsampling path,
            else, the gated signal is from the downsampling path
    @param dropout: specify dropout
    @return: segmentated output, attention coefficient list for each skip connections
    '''
    # downsampling path
    conv1 = conv_block(input_layer, filters=64, kernel_initializer=hn)
    pool1 = pool(conv1)

    conv2 = conv_block(pool1, filters=128, kernel_initializer=hn)
    pool2 = pool(conv2)

    conv3 = conv_block(pool2, filters=256, kernel_initializer=hn)
    pool3 = pool(conv3)

    conv4 = conv_block(pool3, filters=512, kernel_initializer=hn, dropout_rate=dropout)
    pool4 = pool(conv4)

    conv5 = conv_block(pool4, filters=1024, kernel_initializer=hn, dropout_rate=dropout)

    # upsampling path
    att01, grid_att01 = attention_block(conv4, conv5, 512, 'grid_att01')
    up1 = up(conv5, filters=512, merge=att01, kernel_initializer=hn)
    conv6 = conv_block(up1, filters=512, kernel_initializer=hn)

    if attention_mode == 'grid':
        att02, grid_att02 = attention_block(conv3, conv6, 256, 'grid_att02')
    else:
        att02, grid_att02 = attention_block(conv3, conv4, 256, 'grid_att02')
    up2 = up(conv6, filters=256, merge=att02, kernel_initializer=hn)
    conv7 = conv_block(up2, filters=256, kernel_initializer=hn)

    if attention_mode == 'grid':
        att03, grid_att03 = attention_block(conv2, conv7, 128, 'grid_att03')
    else:
        att03, grid_att03 = attention_block(conv2, conv3, 128, 'grid_att03')
    up3 = up(conv7, filters=128, merge=att03, kernel_initializer=hn)
    conv8 = conv_block(up3, filters=128, kernel_initializer=hn)

    if attention_mode == 'grid':
        att04, grid_att04 = attention_block(conv1, conv8, 64, 'grid_att04')
    else:
        att04, grid_att04 = attention_block(conv1, conv2, 64, 'grid_att04')
    up4 = up(conv8, filters=64, merge=att04, kernel_initializer=hn)
    conv9 = conv_block(up4, filters=64, kernel_initializer=hn)

    output_layer = Conv2D(4, (1, 1), activation='softmax')(conv9)
    #attention coefficient
    att_co = [grid_att01, grid_att02, grid_att03, grid_att04]
    return output_layer, att_co


def DeepAttUnet_model(input_layer, attention_mode='grid'):
    '''
    Attention Unet with deep supervision
    @param input_layer: input batched image [b,w,h,c]
    @param attention_mode: choice of attention mode, default 'grid'
            where the gated signal derived from upsampling path,
            else, the gated signal is from the downsampling path
    @param dropout: specify dropout
    @return: segmentated output, attention coefficient list for each skip connections
    '''
    gauss1 = GaussianNoise(0.01)(input_layer)
    # downsampling path
    conv1 = conv_block(gauss1, filters=64, kernel_initializer=hn)
    pool1 = pool(conv1)

    conv2 = conv_block(pool1, filters=128, kernel_initializer=hn)
    pool2 = pool(conv2)

    conv3 = conv_block(pool2, filters=256, kernel_initializer=hn)
    pool3 = pool(conv3)

    conv4 = conv_block(pool3, filters=512, kernel_initializer=hn, dropout_rate=0.3)
    pool4 = pool(conv4)

    conv5 = conv_block(pool4, filters=1024, kernel_initializer=hn, dropout_rate=0.3)

    # upsampling path
    att01, grid_att01 = attention_block(conv4, conv5, 512, 'grid_att01')
    up1 = up(conv5, filters=512, merge=att01, kernel_initializer=hn)
    conv6 = conv_block(up1, filters=512, kernel_initializer=hn)

    if attention_mode == 'grid':
        att02, grid_att02 = attention_block(conv3, conv6, 256, 'grid_att02')
    else:
        att02, grid_att02 = attention_block(conv3, conv4, 256, 'grid_att02')
    up2 = up(conv6, filters=256, merge=att02, kernel_initializer=hn)
    conv7 = conv_block(up2, filters=256, kernel_initializer=hn)
    # injection block 1
    seg01 = Conv2D(4, (1, 1), padding='same')(conv7)
    up_seg01 = UpSampling2D()(seg01)

    if attention_mode == 'grid':
        att03, grid_att03 = attention_block(conv2, conv7, 128, 'grid_att03')
    else:
        att03, grid_att03 = attention_block(conv2, conv3, 128, 'grid_att03')
    up3 = up(conv7, filters=128, merge=att03, kernel_initializer=hn)
    conv8 = conv_block(up3, filters=128, kernel_initializer=hn)
    # injection block 2
    seg02 = Conv2D(4, (1, 1), padding='same')(conv8)
    add_21 = Add()([seg02, up_seg01])
    up_seg02 = UpSampling2D()(add_21)

    if attention_mode == 'grid':
        att04, grid_att04 = attention_block(conv1, conv8, 64, 'grid_att04')
    else:
        att04, grid_att04 = attention_block(conv1, conv2, 64, 'grid_att04')
    up4 = up(conv8, filters=64, merge=att04, kernel_initializer=hn)
    conv9 = conv_block(up4, filters=64, kernel_initializer=hn)
    # injection block 3
    seg03 = Conv2D(4, (1, 1), padding='same')(conv9)
    add_32 = Add()([seg03, up_seg02])

    #segmentated output
    output_layer = Conv2D(4, (1, 1), activation='softmax')(add_32)
    #attention coefficient
    att_co = [grid_att01, grid_att02, grid_att03, grid_att04]

    return output_layer, att_co


def selfGuidedAtt_v02(x):
    '''
    Resnet as backbone for multiscale feature retrieval.
    Each resblock output(input signal), next resblock output(gated signal) is
    feed into the gated attention for multi scale feature refinement.
    Each gated attention output is pass through a bottle neck layer to standardize
    the channel size by squashing them to desired filter size of 64.
    The features are upsampled at each block to the corresponding [wxh] dimension
    of w:240, h:240.
    The upsampled features are concat and squash to corresponding channel size of 64
    which yield multiscale feature.
    :param x: batched images
    :return: feature maps of each res block
    '''
    #inject noise
    gauss1 = GaussianNoise(0.01)(x)
    #retrieve input dimension
    b,w,h,c = x.shape
    #---- ResNet and Multiscale Features----
    #1st block
    conv01 = CoordConv(x_dim=w, y_dim=h, with_r=False, filters=64, strides=(1,1),
                      kernel_size = 3, padding='same', kernel_initializer=hn, name='conv01')(gauss1)
    res_block01 = res_block_sep_v2(conv01, filters=[128, 64], layer_name=["conv02", "conv03", "add01"], dropout_rate=None)
    #2nd block
    down_01 = down_sampling_sep_v2(res_block01, filters=128, layer_name = 'down_01',  kernel_initializer=hn,
                               mode='coord',x_dim=w//2, y_dim=w//2)
    res_block02 = res_block_sep_v2(down_01, filters=[256, 128], layer_name=["conv04", "conv05", "add02"], dropout_rate=None)
    #3rd block
    down_02 = down_sampling_sep_v2(res_block02, filters=256, layer_name = 'down_02',  kernel_initializer=hn,
                               mode='coord',x_dim=w//4, y_dim=h//4)
    res_block03 = res_block_sep_v2(down_02, filters=[512, 256], layer_name=["conv06", "conv07", "add03"], dropout_rate=None)
    #4th block
    down_03 = down_sampling_sep_v2(res_block03, filters=512, layer_name = 'down_03',  kernel_initializer=hn,
                               mode='coord',x_dim=w//8, y_dim=h//8)
    res_block04 = res_block_sep_v2(down_03, filters=[1024, 512], layer_name=["conv08", "conv09", "add04"], dropout_rate=None)
    # *apply activation function for the last output
    res_block04 = PReLU(shared_axes=[1,2])(res_block04)
    #grid attention blocks
    att_block01, g_att01 = attention_block(res_block01,res_block02,64,'grid_att01')
    att_block02, g_att02 = attention_block(res_block02,res_block03,128,'grid_att02')
    att_block03, g_att03 = attention_block(res_block03, res_block04,256,'gird_att03')
    #bottle neck => layer squash all attention block to same filter size 64
    bottle01 = Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer=hn)(att_block01)
    bottle02 = Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer=hn)(att_block02)
    bottle03 = Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer=hn)(att_block03)
    #upsampling for all layers to same (wxh) dimension=>240x240
    up01 = bottle01 #[240,240,64]
    up02 = UpSampling2D(size=(2, 2), interpolation='bilinear')(bottle02) #[120,120,64]=>[240,240,64]
    up03 = UpSampling2D(size=(4,4), interpolation='bilinear')(bottle03) #[60,60,64]=>[240,240,64]
    #multiscale features
    concat_all = concatenate([up01,up02,up03],axis=-1) #[240,240,3*64]
    #squeeze to have the same channel as upsampled features [240,240,3*64] => [240,240,64]
    ms_feature = Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer=hn)(concat_all)
    #Segmentations from multiscale features *without softmax activation
    seg_01 = Conv2D(4, (1,1), name='seg_01')(up01)
    seg_02 = Conv2D(4, (1,1), name='seg_02')(up02)
    seg_03 = Conv2D(4, (1,1), name='seg_03')(up03)

    #----self guided attention blocks-----
    ga_01, f_pc01 = guided_attention(up01, ms_feature, layer_name_ga[0])
    ga_02, f_pc02 = guided_attention(up02, ms_feature, layer_name_ga[1])
    ga_03, f_pc03 = guided_attention(up03, ms_feature, layer_name_ga[2])
    #Segmentations from guided attention features *without softmax activation
    seg_ga01 = Conv2D(4, (1,1), name='seg_ga01')(ga_01)
    seg_ga02 = Conv2D(4, (1,1), name='seg_ga02')(ga_02)
    seg_ga03 = Conv2D(4, (1,1), name='seg_ga03')(ga_03)
    #outputs for xent losses
    output_xent = [seg_01, seg_02, seg_03, seg_ga01, seg_ga02, seg_ga03]
    #output for dice coefficient loss
    pred_seg = Add()(output_xent)
    output_dice = Softmax()(pred_seg/len(output_xent))
    #output for feature visualization
    #gated attention
    gated_attention = [g_att01, g_att02, g_att03]
    #pam and cam features
    f_pc = [f_pc01, f_pc02, f_pc03]
    return output_xent, output_dice, gated_attention, f_pc

def selfGuidedAtt_v01(x):
    '''
    Resnet as backbone for multiscale feature retrieval.
    Each resblock output(input signal), next resblock output(gated signal) is
    feed into the gated attention for multi scale feature refinement.
    Each gated attention output is pass through a bottle neck layer to standardize
    the channel size by squashing them to desired filter size of 64.
    The features are upsampled at each block to the corresponding [wxh] dimension
    of w:240, h:240.
    The upsampled features are concat and squash to corresponding channel size of 64
    which yield multiscale feature.
    :param x: batched images
    :return: feature maps of each res block
    '''
    #inject noise
    gauss1 = GaussianNoise(0.01)(x)
    #---- ResNet and Multiscale Features----
    #1st block
    conv01 = CoordConv(x_dim=240, y_dim=240, with_r=False, filters=64, strides=(1,1),
                      kernel_size = 3, padding='same', kernel_initializer=hn, name='conv01')(gauss1)
    res_block01 = res_block_sep(conv01, filters=[128, 64], layer_name=["conv02", "conv03", "add01"])
    #2nd block
    down_01 = down_sampling_sep(res_block01, filters=128, layer_name = 'down_01',  kernel_initializer=hn,
                               mode='normal',x_dim=120, y_dim=120)
    res_block02 = res_block_sep(down_01, filters=[256, 128], layer_name=["conv04", "conv05", "add02"])
    #3rd block
    down_02 = down_sampling_sep(res_block02, filters=256, layer_name = 'down_02',  kernel_initializer=hn,
                               mode='normal',x_dim=60, y_dim=60)
    res_block03 = res_block_sep(down_02, filters=[512, 256], layer_name=["conv06", "conv07", "add03"])
    #4th block
    down_03 = down_sampling_sep(res_block03, filters=512, layer_name = 'down_03',  kernel_initializer=hn,
                               mode='normal',x_dim=30, y_dim=30)
    res_block04 = res_block_sep(down_03, filters=[1024, 512], layer_name=["conv08", "conv09", "add04"])
    #grid attention blocks
    att_block01, g_att01 = attention_block(res_block01,res_block02,64,'grid_att01')
    att_block02, g_att02 = attention_block(res_block02,res_block03,128,'grid_att02')
    att_block03, g_att03 = attention_block(res_block03, res_block04,256,'gird_att03')
    #bottle neck => layer squash all attention block to same filter size 64
    bottle01 = Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer=hn)(att_block01)
    bottle02 = Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer=hn)(att_block02)
    bottle03 = Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer=hn)(att_block03)
    #upsampling for all layers to same (wxh) dimension=>240x240
    up01 = bottle01 #[240,240,64]
    up02 = UpSampling2D(size=(2, 2), interpolation='bilinear')(bottle02) #[120,120,64]=>[240,240,64]
    up03 = UpSampling2D(size=(4,4), interpolation='bilinear')(bottle03) #[60,60,64]=>[240,240,64]
    #multiscale features
    concat_all = concatenate([up01,up02,up03],axis=-1) #[240,240,3*64]
    #squeeze to have the same channel as upsampled features [240,240,3*64] => [240,240,64]
    ms_feature = Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer=hn)(concat_all)
    #Segmentations from multiscale features *without softmax activation
    seg_01 = Conv2D(4, (1,1), name='seg_01')(up01)
    seg_02 = Conv2D(4, (1,1), name='seg_02')(up02)
    seg_03 = Conv2D(4, (1,1), name='seg_03')(up02)

    #----self guided attention blocks-----
    ga_01, f_pc01 = guided_attention(up01, ms_feature, layer_name_ga[0])
    ga_02, f_pc02 = guided_attention(up02, ms_feature, layer_name_ga[1])
    ga_03, f_pc03 = guided_attention(up03, ms_feature, layer_name_ga[2])
    #Segmentations from guided attention features *without softmax activation
    seg_ga01 = Conv2D(4, (1,1), name='seg_ga01')(ga_01)
    seg_ga02 = Conv2D(4, (1,1), name='seg_ga02')(ga_02)
    seg_ga03 = Conv2D(4, (1,1), name='seg_ga03')(ga_03)
    #outputs for xent losses
    output_xent = [seg_01, seg_02, seg_03, seg_ga01, seg_ga02, seg_ga03]
    #output for dice coefficient loss
    pred_seg = Add()(output_xent)
    output_dice = Softmax()(pred_seg/len(output_xent))
    #output for feature visualization
    #gated attention
    gated_attention = [g_att01, g_att02, g_att03]
    #pam and cam features
    f_pc = [f_pc01, f_pc02, f_pc03]
    return output_xent, output_dice, gated_attention, f_pc

