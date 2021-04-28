import tensorflow as tf
from coord_conv import CoordConv
from tensorflow.keras.layers import Conv2D, UpSampling2D, Activation, Add, Multiply, MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, Dropout
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.layers import ReLU, PReLU
from tensorflow.keras.layers import Conv3D, UpSampling3D, MaxPool3D


hn = 'he_normal' #kernel initializer

def conv_block(x_in, filters, batch_norm=False, kernel_size=(3, 3),
               kernel_initializer='glorot_uniform', acti='relu', dropout_rate=None):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x_in)
    # if batch_norm == True:
    #     x = BatchNormalization()(x)
    x = Activation(acti)(x)
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x


def coordconv_block(x_in, x_dim, y_dim, filters, batch_norm=False, kernel_size=(3, 3),
                    kernel_initializer='glorot_uniform', acti='relu', dropout_rate=None, with_r=False):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x_in)
    # if batch_norm == True:
    #     x = BatchNormalization()(x)
    x = Activation(acti)(x)
    x = CoordConv(x_dim, y_dim, with_r, filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x


def conv_2d(x_in, filters, batch_norm=False, kernel_size=(3, 3), acti='relu',
            kernel_initializer='glorot_uniform', dropout_rate=None):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x


def pool(x_in, pool_size=(2, 2), type='Max'):
    if type == 'Max':
        p = MaxPooling2D(pool_size)(x_in)
    return p


def up(x_in, filters, merge, batch_norm=False,
       kernel_initializer='glorot_uniform', dropout_rate=None, size=(2, 2)):
    u = UpSampling2D(size)(x_in)
    conv = conv_2d(u, filters, batch_norm, acti='relu', kernel_initializer=kernel_initializer,
                   dropout_rate=dropout_rate)
    concat = tf.concat([merge, conv], axis=-1)
    return concat

def attention_block(input_signal, gated_signal, filters):
    #input signal feature maps
    is_fm = Conv2D(filters, kernel_size=(1,1), strides=(2, 2), padding = 'same')(input_signal)
    #gated signal feature maps
    gs_fm = Conv2D(filters, kernel_size=(1,1), strides=(1, 1), padding = 'same')(gated_signal)
    #debugger
    assert is_fm.shape!=gs_fm.shape, "Feature maps shape doesn't match!"
    #element wise sum
    add = Add()([is_fm, gs_fm])
    acti = Activation('relu')(add)
    #downsampled attention coefficient
    bottle_neck = Conv2D(1, kernel_size=(1,1), activation='sigmoid')(acti)
    #bilinear interpolation to get attention coeffcient
    alpha = UpSampling2D(interpolation='bilinear')(bottle_neck)
    #filter off input signal's features with attention coefficient
    multi = Multiply()([input_signal, alpha])
    return multi

def conv_block_sep(x_in, filters, layer_name, batch_norm=False, kernel_size=(3, 3),
               kernel_initializer='glorot_uniform', acti='relu', dropout_rate=None):
    assert type(filters)==list, "Please input filters of type list."
    assert type(layer_name)==list, "Please input filters of type list."
    x = SeparableConv2D(filters[0], kernel_size, padding='same', kernel_initializer=kernel_initializer, name = layer_name[0])(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    x = SeparableConv2D(filters[1], kernel_size, padding='same', kernel_initializer=kernel_initializer, name = layer_name[1])(x)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x

def conv_2d_sep(x_in, filters, layer_name, batch_norm=False, kernel_size=(3, 3), acti='relu',
            kernel_initializer='glorot_uniform', dropout_rate=None):
    x = SeparableConv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer, name=layer_name)(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x

def conv_2d(x_in, filters, layer_name, strides=(1,1), batch_norm=False, kernel_size=(3, 3), acti='relu',
            kernel_initializer='glorot_uniform', dropout_rate=None):
    x = Conv2D(filters, kernel_size, strides, padding='same', kernel_initializer=kernel_initializer, name=layer_name)(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x

def down_sampling_sep(x_in, filters, layer_name, batch_norm=False, kernel_size=(3, 3), acti='relu',
            kernel_initializer='glorot_uniform', dropout_rate=None, mode ='coord', x_dim=None, y_dim=None):
    assert mode=='coord' or mode=='normal', "Use 'coord' or 'normal' for mode!"
    if mode=='coord':
        #seperable coordconv
        assert (x_dim!=None and y_dim!=None), "Please input dimension for CoordConv!"
        x = Conv2D(1, kernel_size, strides=(2, 2), padding='same', kernel_initializer=kernel_initializer)(x_in)
        x = CoordConv(x_dim=x_dim, y_dim=y_dim, with_r=False, filters=filters, strides=(1,1),
                      kernel_size = 3, padding='same', kernel_initializer=kernel_initializer, name=layer_name)(x)
    else:
        #normal mode
        x = SeparableConv2D(filters, kernel_size, strides=(2, 2), padding='same', kernel_initializer=kernel_initializer, name=layer_name)(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x

def res_block_sep(x_in, filters,  layer_name, batch_norm=False, kernel_size=(3, 3),
               kernel_initializer='glorot_uniform', acti='relu', dropout_rate=None):
    assert len(filters)==2, "Please assure that there is 3 values for filters."
    assert len(layer_name)==3, "Please assure that there is 3 values for layer name"
    layer_name_conv = [layer_name[i] for i in range(len(layer_name)-1)]
    output_conv_block = conv_block_sep(x_in, filters, layer_name_conv, batch_norm=batch_norm, kernel_size=kernel_size,
                                   kernel_initializer = kernel_initializer, acti = acti, dropout_rate=dropout_rate)
    output_add = Add(name = layer_name[-1])([output_conv_block, x_in])
    return output_add


def conv_block_sep_v2(x, filters, layer_name, norm_fn='bn', kernel_size=(3, 3),
               kernel_initializer='glorot_uniform', acti_fn='relu', dropout_rate=None):
    '''
    Dual convolution block with [full pre-activation], Norm -> Acti -> Conv
    :param x: Input features
    :param filters: A list that contains the number of filters for 1st and 2nd convolutional layer
    :param layer_name: A list  that contains the name for the 1st and 2nd convolutional layer
    :param norm_fn: Tensorflow function for normalization, 'bn' for Batch Norm, 'gn' for Group Norm
    :param kernel_size: Kernel size for both convolutional layer with 3x3 as default
    :param kernel_initializer: Initializer for kernel weights with 'glorot uniform' as default
    :param acti_fn: Tensorflow function for activation, 'relu' for ReLU, 'prelu' for PReLU
    :param dropout_rate: Specify dropouts for layers
    :return: Feature maps of same size as input with number of filters equivalent to the last layer
    '''
    assert type(filters)==list, "Please input filters of type list."
    assert type(layer_name)==list, "Please input filters of type list."
    assert acti_fn!= None, 'There should be an activation functino specified'
    #1st convolutional block
    if norm_fn=='bn':
        x = BatchNormalization()(x)
    elif norm_fn=='gn':
        x = GroupNormalization()(x)
    if acti_fn=='relu':
        x = ReLU()(x)
    elif acti_fn=='prelu':
        x = PReLU(shared_axes=[1,2])(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    x = SeparableConv2D(filters[0], kernel_size, padding='same', kernel_initializer=kernel_initializer, name = layer_name[0])(x)
    #2nd convolutional block
    if norm_fn=='bn':
        x = BatchNormalization()(x)
    elif norm_fn=='gn':
        x = GroupNormalization()(x)
    if acti_fn=='relu':
        x = ReLU()(x)
    elif acti_fn=='prelu':
        x = PReLU(shared_axes=[1,2])(x)
    x = SeparableConv2D(filters[1], kernel_size, padding='same', kernel_initializer=kernel_initializer, name = layer_name[1])(x)
    return x


def down_sampling_sep_v2(x, filters, layer_name, norm_fn='bn', kernel_size=(3, 3), acti_fn='relu',
            kernel_initializer='glorot_uniform', dropout_rate=None, mode ='coord', x_dim=None, y_dim=None):
    '''
    Down sampling function version 2 with Convolutional layer of stride 2 as downsampling operation, with
    [full pre-activation], Norm -> Acti -> Conv
    :param x: Input features
    :param filters: Number of filters for Convolutional layer of stride 2
    :param layer_name: Layer name for convolutional layer
    :param norm_fn: Tensorflow function for normalization, 'bn' for Batch Norm, 'gn' for Group Norm
    :param kernel_size: Kernel size for both convolutional layer with 3x3 as default
    :param acti_fn: Tensorflow function for activation, 'relu' for ReLU, 'prelu' for PReLU
    :param kernel_initializer: Initializer for kernel weights with 'glorot uniform' as default
    :param dropout_rate: Specify dropouts for layers
    :param mode: 'coord' for Seperable Coord Conv, 'normal' for Seperable Conv
    :param x_dim: x dimension for coord conv
    :param y_dim: y dimension for coord conv
    :return: Feature maps of size scaled down by 2 with number of filters specified
    '''
    assert mode=='coord' or mode=='normal',  "Use 'coord' or 'normal' for mode!"
    assert acti_fn!= None, 'There should be an activation functino specified'
    #normalization
    if norm_fn=='bn':
        x = BatchNormalization()(x)
    elif norm_fn=='gn':
        x = GroupNormalization()(x)
    if acti_fn=='relu':
        x = ReLU()(x)
    #activation
    elif acti_fn=='prelu':
        x = PReLU(shared_axes=[1,2])(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    if mode=='coord':
        #seperable coordconv
        assert (x_dim!=None and y_dim!=None), "Please input dimension for CoordConv!"
        x = Conv2D(1, kernel_size, strides=(2, 2), padding='same', kernel_initializer=kernel_initializer)(x)
        x = CoordConv(x_dim=x_dim, y_dim=y_dim, with_r=False, filters=filters, strides=(1,1),
                      kernel_size = 3, padding='same', kernel_initializer=kernel_initializer, name=layer_name)(x)
    else:
        #normal mode
        x = SeparableConv2D(filters, kernel_size, strides=(2, 2), padding='same', kernel_initializer=kernel_initializer, name=layer_name)(x)
    return x

def res_block_sep_v2(x_in, filters,  layer_name, norm_fn='gn', kernel_size=(3, 3),
               kernel_initializer='glorot_uniform', acti_fn='prelu', dropout_rate=None):
    assert len(filters)==2, "Please assure that there is 2 values for filters."
    assert len(layer_name)==3, "Please assure that there is 3 values for layer name"
    layer_name_conv = [layer_name[i] for i in range(len(layer_name)-1)]
    output_conv_block = conv_block_sep_v2(x_in, filters, layer_name_conv, norm_fn=norm_fn, kernel_size=kernel_size,
                                   kernel_initializer = kernel_initializer, acti_fn = acti_fn, dropout_rate=dropout_rate)
    output_add = Add(name = layer_name[-1])([output_conv_block, x_in])
    return output_add

#------------------3D Section------------------------------------------


def conv_block_3D(x, filters, norm_fn='gn', kernel_size=3,
               kernel_initializer=hn, acti_fn='prelu', dropout_rate=None):
    '''
    Dual convolution block with [full pre-activation], Norm -> Acti -> Conv
    :param x: Input features
    :param filters: A list that contains the number of filters for 1st and 2nd convolutional layer
    :param norm_fn: Tensorflow function for normalization, 'bn' for Batch Norm, 'gn' for Group Norm
    :param kernel_size: Kernel size for both convolutional layer with 3x3 as default
    :param kernel_initializer: Initializer for kernel weights with 'glorot uniform' as default
    :param acti_fn: Tensorflow function for activation, 'relu' for ReLU, 'prelu' for PReLU
    :param dropout_rate: Specify dropouts for layers
    :return: Feature maps of same size as input with number of filters equivalent to the last layer
    '''
    assert type(filters)==list, "Please input filters of type list."
    assert acti_fn!= None, 'There should be an activation function specified'
    #1st convolutional block
    if norm_fn=='bn':
        x = BatchNormalization()(x)
    elif norm_fn=='gn':
        x = GroupNormalization()(x)
    if acti_fn=='relu':
        x = ReLU()(x)
    elif acti_fn=='prelu':
        x = PReLU(shared_axes=[1,2,3])(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    x = Conv3D(filters[0], kernel_size, padding='same', kernel_initializer=kernel_initializer)(x)
    #2nd convolutional block
    if norm_fn=='bn':
        x = BatchNormalization()(x)
    elif norm_fn=='gn':
        x = GroupNormalization()(x)
    if acti_fn=='relu':
        x = ReLU()(x)
    elif acti_fn=='prelu':
        x = PReLU(shared_axes=[1,2,3])(x)
    x = Conv3D(filters[1], kernel_size, padding='same', kernel_initializer=kernel_initializer)(x)
    return x


def down_sampling_3D(x, filters, norm_fn='gn', kernel_size=3, acti_fn='relu',
            kernel_initializer=hn, dropout_rate=None):
    '''
    Down sampling function version 2 with Convolutional layer of stride 2 as downsampling operation, with
    [full pre-activation], Norm -> Acti -> Conv
    :param x: Input features
    :param filters: Number of filters for Convolutional layer of stride 2
    :param norm_fn: Tensorflow function for normalization, 'bn' for Batch Norm, 'gn' for Group Norm
    :param kernel_size: Kernel size for both convolutional layer with 3x3 as default
    :param acti_fn: Tensorflow function for activation, 'relu' for ReLU, 'prelu' for PReLU
    :param kernel_initializer: Initializer for kernel weights with 'glorot uniform' as default
    :param dropout_rate: Specify dropouts for layers
    :return: Feature maps of size scaled down by 2 with number of filters specified
    '''
    assert acti_fn!= None, 'There should be an activation function specified'
    #normalization
    if norm_fn=='bn':
        x = BatchNormalization()(x)
    elif norm_fn=='gn':
        x = GroupNormalization()(x)
    if acti_fn=='relu':
        x = ReLU()(x)
    #activation
    elif acti_fn=='prelu':
        x = PReLU(shared_axes=[1,2,3])(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    #normal mode
    x = Conv3D(filters, kernel_size, strides=(2,2,2), padding='same', kernel_initializer=kernel_initializer)(x)
    return x


def res_block_3D(x_in, filters, norm_fn='gn', kernel_size=3,
               kernel_initializer=hn, acti_fn='prelu', dropout_rate=None):
    '''
    This function construct the residual block in 3D by input->conv_block_3D->concat([input,conv_output])
    :param x: Input features
    :param filters: A list that contains the number of filters for 1st and 2nd convolutional layer
    :param norm_fn: Tensorflow function for normalization, 'bn' for Batch Norm, 'gn' for Group Norm
    :param kernel_size: Kernel size for both convolutional layer with 3x3 as default
    :param kernel_initializer: Initializer for kernel weights with 'glorot uniform' as default
    :param acti_fn: Tensorflow function for activation, 'relu' for ReLU, 'prelu' for PReLU
    :param dropout_rate: Specify dropouts for layers
    :return: Resblock output => concatenating input with 2*convlutional output
    '''
    assert len(filters)==2, "Please assure that there is 2 values for filters."
    output_conv_block = conv_block_3D(x_in, filters, norm_fn=norm_fn, kernel_size=kernel_size,
                                   kernel_initializer = kernel_initializer, acti_fn = acti_fn, dropout_rate=dropout_rate)
    output_add = Add()([output_conv_block, x_in])
    return output_add


def up_3D(x_in, filters, merge, kernel_initializer=hn, size=(2, 2, 2)):
    '''
    This function carry out the operation of deconvolution => upsampling + convolution, and
    concatenating feture maps from the skip connection with the deconv feature maps
    @param x_in: input feature
    @param filters: Number of filters
    @param merge: featrure maps from the skip connection
    @param kernel_initializer: Initializer for kernel weights with 'glorot uniform' as default
    @param size: Upsampling size, by default (1,2,2)
    @return: concatenate feature maps of skip connection output and upsampled feature maps from previous output
    '''
    u = UpSampling3D(size)(x_in)
    conv = Conv3D(filters=filters, kernel_size=3, padding='same',
                  kernel_initializer=kernel_initializer)(u)
    conv = PReLU(shared_axes=[1,2,3])(conv)
    concat = tf.concat([merge, conv], axis=-1)
    return concat
