import numpy as np
import random as rn
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.metrics import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import keras.losses as KLoss
import keras as K
#from keras import backend as keras
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as keras
from keras.losses import categorical_crossentropy

MySeed = 42

def Accumelated_Tversky(y_true , y_pred):

    Ncl = y_pred.shape[-1]
    keras_w = get_weights(y_true, y_pred)
    Acc_Value = []


    for L in range(Ncl):
        Tv_Value = Single_Tversky(y_true[:,:,:,:,L] , y_pred[:,:,:,:,L],keras_w[L])
        Acc_Value.append(Tv_Value)


    K_value_List = keras.stack(Acc_Value)


    # K_value = keras.sum(K_value_List)

    if(Ncl == 13):
        T = 13 - keras.sum(K_value_List)
    elif(Ncl == 14):
        T = 14 - keras.sum(K_value_List)
    elif (Ncl == 2):
        T = 2 - keras.sum(K_value_List)
    elif (Ncl == 1):
        T = 1 - keras.sum(K_value_List)

    return T

def Single_Tversky(y_true , y_pred, keras_w):

    alpha = 0.5
    beta = 0.5

    ones = keras.ones(keras.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true


    num = keras.sum(p0 * g0)
    den = (num + alpha * keras.sum(p0 * g1) + beta * keras.sum(p1 * g0))

    Tv_Value = keras_w * num / den

    return Tv_Value    #num,den

def Tversky_Value(y_true , y_pred):
    # Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
    # -> the score is computed for each class separately and then summed
    # alpha=beta=0.5 : dice coefficient
    # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    # alpha+beta=1   : produces set of F*-scores
    # implemented by E. Moebel, 06/04/18

    alpha = 0.5
    beta = 0.5

    # Round_y_pred = tf.round(y_pred)

    ones = keras.ones(keras.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    keras_w = get_weights(y_true, y_pred) # kilany implementation

    num = keras.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * keras.sum(p0 * g1, (0, 1, 2, 3)) + beta * keras.sum(p1 * g0, (0, 1, 2, 3))

    T = keras.sum(keras_w * (num / den)) #keras.sum((num / den))   # when summing over classes, T has dynamic range [0 Ncl]
    # T = keras.mean(keras_w * (num / den))  # kilany implementation

    Ncl = keras.cast(keras.shape(y_true)[-1], 'float32') # commented in kilany implementation
    print("T shape in Tversky_Value = " + str(keras.shape(num)))
    return T #Ncl - T
    # return 1-T
#
def get_weights(y_true,y_pred):

    Ncl = y_pred.shape[-1]
    print("Ncl after lastdimension = " + str(Ncl))

    counts = keras.sum(y_true,axis=(0,1,2,3))
    Total  = keras.sum(counts)
    w = 1 - (counts / Total)


    print("The keras weights shape= ")
    print(str(w) )
    return w

def generalized_dice_loss_w(y_true, y_pred):
    ## Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    ## Source of code: https://github.com/keras-team/keras/issues/9395

    keras_w = get_weights(y_true,y_pred)
    ## Round_y_pred = keras.round(y_pred)

    ## Compute gen dice coef:
    numerator = y_true * y_pred
    numerator = keras.sum(numerator, (0, 1, 2, 3)) #keras_w * keras.sum(numerator, (0, 1, 2, 3))
    print("The numerator = " + str(numerator))
    ## numerator = keras.sum(numerator)  # commented on 08/06/2020

    denominator = y_true + y_pred
    denominator = keras.sum(denominator, (0, 1, 2, 3)) #keras_w * keras.sum(denominator, (0, 1, 2, 3))
    print("The denominator = " + str(denominator))
    ## denominator = keras.sum(denominator)  # commented on 08/06/2020

    # gen_dice_coef = keras.sum(2 * numerator / denominator)
    gen_dice_coef = keras.sum(2 * keras_w * numerator / denominator)

    return  gen_dice_coef

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    print("Size of y_true = " + str(keras.get_variable_shape(y_pred)))
    #y_pred_np = keras.eval(y_pred)
    LabelChannel = 1#13#y_pred_np.shape[-1] #13 # including the background
    #y_pred_np = np.argmax(y_pred_np, axis = -1)
    #y_pred = keras.variable( K.utils.to_categorical(y_pred_np,LabelChannel) )

    DiceList = []

    if(LabelChannel >= 2):
        for L in range(LabelChannel):
            if (L > 0):
                intersection = keras.sum(keras.abs(y_true[:, :, :, :, L] * y_pred[:, :, :, :, L]))
                Sum_y_true = keras.sum(y_true[:, :, :, :, L])
                Sum_y_pred = keras.sum(y_pred[:, :, :, :, L])
                Dice = (2 * intersection + smooth) / (Sum_y_true + Sum_y_pred + smooth)
                DiceList.append(Dice)
        K_DiceList = keras.stack(DiceList)

    else:
        intersection = keras.sum(keras.abs(y_true * y_pred))
        Sum_y_true = keras.sum(y_true)
        Sum_y_pred = keras.sum(y_pred)
        Dice = (2 * intersection + smooth) / (Sum_y_true + Sum_y_pred + smooth)
        DiceList.append(Dice)
        K_DiceList = keras.stack(DiceList)


    K_Scaler_DiceList = keras.mean(K_DiceList)
    return K_Scaler_DiceList

    # intersection = keras.sum(keras.abs(y_true[:, :, :, 1:] * y_pred[:, :, :, 1:]), axis=-1)
    # return (2. * intersection + smooth) / (keras.sum(keras.square(y_true[:, :, :, 1:]), -1) + keras.sum(keras.square(y_pred[:, :, :, 1:]), -1) + smooth)



def dice_loss(y_true, y_pred):
    #Dice = dice_coef(y_true, y_pred)
    Dice = generalized_dice_loss_w(y_true, y_pred)
    Ncl = y_pred.shape[-1]
    if(Ncl == 13):
        return 13-Dice
    elif(Ncl == 14):
        return 14-Dice
    elif (Ncl == 2):
        return 2 - Dice
    else:#elif (Ncl == 1):
        return 1 - Dice

def Tversky_loss(y_true, y_pred):

    Tversky = Tversky_Value(y_true, y_pred)
    Ncl = y_pred.shape[-1]
    if (Ncl == 13):
        return 13-Tversky
    elif (Ncl == 14):
        return 14-Tversky
    elif (Ncl == 2):
        return 2-Tversky
    else:#elif (Ncl == 1):
        return 1-Tversky


def binary_crossEntropy_Dice_loss(y_true, y_pred):
    return 0.5*KLoss.binary_crossentropy(y_true, y_pred)+0.5*dice_loss(y_true, y_pred)

def Encode_Conv_Block(InputLayer, filters_no, stage, block, filter_Size = 3, strides_size = 1):

    conv_name_base = 'conv_' + str(stage) + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'

    tuple_filter_size = (filter_Size, filter_Size, filter_Size)
    tuple_strides_size = (strides_size, strides_size, strides_size)

    K_Initializer = K.initializers.he_normal(seed = MySeed)

    conv1 = Conv3D(filters_no, tuple_filter_size, dilation_rate = tuple_strides_size, padding='same', name = conv_name_base + '_E1', kernel_initializer = K_Initializer)(InputLayer) #'he_normal'
    Batch1 = BatchNormalization(axis=-1 , name = bn_name_base + '_E1')(conv1)
    Activate1 = Activation('relu')(Batch1)

    conv2 = Conv3D(filters_no, tuple_filter_size, dilation_rate = tuple_strides_size, padding='same', name = conv_name_base + '_E2' , kernel_initializer = K_Initializer)(Activate1) #'he_normal'
    Batch2 = BatchNormalization(axis=-1 , name = bn_name_base + '_E2')(conv2)
    Activate2 = Activation('relu')(Batch2)

    return Activate2

def Decoder_Conv_Block(Input_For_Upsampling,Input_For_Concatenate,filters_no, stage, block,BatchFlag, filter_Size = 3, strides_size = 1):

    conv_name_base = 'conv_' + str(stage) + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'

    tuple_filter_size = (filter_Size, filter_Size, filter_Size)
    tuple_strides_size = (strides_size, strides_size, strides_size)

    K_Initializer = K.initializers.he_normal(seed=MySeed)

    UpSampling = UpSampling3D(size=(2, 2, 2))(Input_For_Upsampling)
    conv1 = Conv3D(filters_no, tuple_filter_size, dilation_rate = tuple_strides_size, activation='relu', padding='same', name = conv_name_base + '_D00', kernel_initializer = K_Initializer)(UpSampling) #'he_normal'

    if(BatchFlag == 0):
        merge = concatenate([Input_For_Concatenate, conv1], axis=-1)  # check if it should be axis = 4
    else:
        Batch_1 = BatchNormalization(axis=-1 , name = bn_name_base + '_D00')(conv1)
        Activation_1 = Activation('relu')(Batch_1)
        merge = concatenate([Input_For_Concatenate, Activation_1], axis=-1)  # check if it should be axis = 4

    Final_Activation = Encode_Conv_Block(merge, filters_no, stage, block, filter_Size, strides_size)

    return Final_Activation


def unet(pretrained_weights = None , No_Channels = 2, No_Classes = 13, filter_Size = 3, strides_size = 1, ChoosedLoss = "Dice", BottleNeck = 13):  #  (96,96,80,2)  input_size = (192,192,128,1)


    np.random.seed(MySeed)
    rn.seed(MySeed)
    tf.compat.v1.random.set_random_seed(MySeed)  # tf.random.set_random_seed(MySeed)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    keras.set_session(sess)

    tf.keras.mixed_precision.experimental.set_policy('infer')  #try 'infer_with_float32_vars'

    input_size = (None, None, None, No_Channels)
    input = Input(input_size)

    ## 1- Encoding branch
    conv1 = Encode_Conv_Block(input, 16, 1, "encoder_10", filter_Size, strides_size)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Encode_Conv_Block(pool1, 32, 2, "encoder_20", filter_Size, strides_size)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Encode_Conv_Block(pool2, 64, 3, "encoder_30", filter_Size, strides_size)
    drop3 = conv3 #Dropout(rate = 1 - 0.5, seed = MySeed)(conv3) # temp remove 9/9/2020
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(drop3)

    conv4 = Encode_Conv_Block(pool3, 128, 4, "encoder_40", filter_Size, strides_size)
    drop4 = conv4 #Dropout(rate = 1 - 0.5, seed = MySeed)(conv4) # temp remove 9/9/2020
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Encode_Conv_Block(pool4, 256, 5, "encoder_50", filter_Size, strides_size)
    # conv5 = Encode_Conv_Block(pool3, 256, 5, "encoder_50", filter_Size, strides_size) # not full unet
    drop5 = conv5 #Dropout(rate = 1 - 0.5, seed = MySeed)(conv5) # temp remove 9/9/2020


    ## 2- Decoding branch

    conv6 = Decoder_Conv_Block(drop5, drop4, 128, 4, "decoder_40", 1, filter_Size, strides_size) #conv6 = Decoder_Conv_Block(drop5, drop4, 128, 4, "decoder_40", 0, filter_Size, strides_size)

    conv7 = Decoder_Conv_Block(conv6, conv3, 64, 3, "decoder_30", 1, filter_Size, strides_size)
    # conv7 = Decoder_Conv_Block(drop5, drop3, 64, 3, "decoder_30", 1, filter_Size, strides_size) # not full unet

    conv8 = Decoder_Conv_Block(conv7, conv2, 32, 2, "decoder_20", 1, filter_Size, strides_size)

    conv9 = Decoder_Conv_Block(conv8, conv1, 16, 1, "decoder_10", 1, filter_Size, strides_size)

    tuple_filter_size = (filter_Size, filter_Size, filter_Size)
    tuple_strides_size= (strides_size, strides_size, strides_size)

    K_Initializer = K.initializers.he_normal(seed=MySeed)

    conv9 = Conv3D(BottleNeck, tuple_filter_size, dilation_rate = tuple_strides_size, padding = 'same', kernel_initializer = K_Initializer)(conv9)  #'he_normal'
    Batch9 = BatchNormalization(axis = -1)(conv9)
    conv9 = Activation('relu')(Batch9)

    conv10 = Conv3D(No_Classes, 1, activation = 'sigmoid')(conv9)  #  softmax 'sigmoid' 2-> number of classes

    model = Model(inputs = input, outputs = conv10)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy']) # in literature loss = Dice loss function
    # model.compile(optimizer=Adam(lr=1e-4), loss=categorical_crossentropy, metrics=['categorical_accuracy'])
    # later will use optimizer = 'sgd'
    if(ChoosedLoss == "Dice"):
        model.compile(optimizer=Adam(lr=1e-4), loss=dice_loss, metrics=[generalized_dice_loss_w]) #[dice_coef]
    elif(ChoosedLoss == "Tversky_4"):
        model.compile(optimizer=Adam(lr=1e-4), loss=Tversky_loss, metrics=[Tversky_Value])
    elif (ChoosedLoss == "Tversky_2"):
        model.compile(optimizer=Adam(lr=1e-2), loss=Accumelated_Tversky, metrics=[Accumelated_Tversky])
    elif(ChoosedLoss == "SparseCategoricalCrossEntropy"):
        model.compile(optimizer=Adam(lr=1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    elif (ChoosedLoss == "CategoricalCrossEntropy"):
        model.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    elif(ChoosedLoss == "BinaryCrossEntropy"):
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy',dice_coef])  #'categorical_crossentropy'
    
    model.summary()


    return model


