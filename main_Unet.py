from model_3DUnet_BatchNormalization_nChannels_12Objects import *
from data import *
import os
# import tensorflow as tf

path_to_train    = "/exports/lkeb-hpc/mkmahassan/Data/Becker_UpperLeg/Fold1/Undersampled/Training/"
path_to_Validate = "/exports/lkeb-hpc/mkmahassan/Data/Becker_UpperLeg/Fold1/Undersampled/Validation/"
path_to_test     = "/exports/lkeb-hpc/mkmahassan/Data/Becker_UpperLeg/Fold1/Undersampled/Test/"
path_to_saved_model = "/exports/lkeb-hpc/mkmahassan/Data/Becker_UpperLeg/Fold1/CreatedModels/"
path_to_Weights  = "/exports/lkeb-hpc/mkmahassan/project/Becker_UpperLeg/Weights/"

#1- load data
print("Load training")
Fat_tr  , Water_tr  , FatFraction_tr  , IP_tr  , OP_tr  , Labels_tr   = loadSinglenpyData(path_to_train)
print ("Load validation")
Fat_val , Water_val , FatFraction_val , IP_val , OP_val , Labels_val  = loadSinglenpyData(path_to_Validate)
print ("Load testing")
Fat_test, Water_test, FatFraction_test, IP_test, OP_test, Labels_test = loadSinglenpyData(path_to_test)




No_Of_Classes = Labels_tr.shape[-1]
print("No_of_Classes in the main function = " +str(No_Of_Classes))
BatchSize = 1
NoEpochs = 10#150#200#270 #125
Last_Layer = 'softmax'
# UsedLoss = 'Weighted_12FGAndBG_GeneralizedDice'
Metric = 'accuracy'

#---------------------------------------------------
#2- Create Model_FatWater
#2.1- concatenate training data
inShape = Fat_tr.shape
Fat_tr   = np.reshape(Fat_tr  , [inShape[0], inShape[1], inShape[2], inShape[3], 1])
Water_tr = np.reshape(Water_tr, [inShape[0], inShape[1], inShape[2], inShape[3], 1])

Data_FW = np.concatenate((Fat_tr, Water_tr), axis=-1)


#2.2- concatenate validation data
inShape = Fat_val.shape
Fat_val   = np.reshape(Fat_val  , [inShape[0], inShape[1], inShape[2], inShape[3], 1])
Water_val = np.reshape(Water_val, [inShape[0], inShape[1], inShape[2], inShape[3], 1])

x_val_FW = np.concatenate((Fat_val, Water_val), axis=-1)


#2.3- concatenate testing data
inShape = Fat_test.shape
Fat_test   = np.reshape(Fat_test  , [inShape[0], inShape[1], inShape[2], inShape[3], 1])
Water_test = np.reshape(Water_test, [inShape[0], inShape[1], inShape[2], inShape[3], 1])

x_test_FW = np.concatenate((Fat_test, Water_test), axis=-1)


#4- Create Model_FatWaterIPOP
#4.1- concatenate training data
inShape = IP_tr.shape
IP_tr   = np.reshape(IP_tr  , [inShape[0], inShape[1], inShape[2], inShape[3], 1])
OP_tr   = np.reshape(OP_tr  , [inShape[0], inShape[1], inShape[2], inShape[3], 1])

Data_FWIP   = np.concatenate((Data_FW, IP_tr), axis=-1)
Data_FWIPOP = np.concatenate((Data_FWIP, OP_tr), axis=-1)

# TrainTest = Data_FWIPOP[0:3,:,:,:,:]
# # TShape = TrainTest.shape
# # TrainTest = np.reshape(TrainTest, [1, TShape[0], TShape[1], TShape[2], TShape[3]])


#4.2- concatenate validation data
inShape = IP_val.shape
IP_val  = np.reshape(IP_val  , [inShape[0], inShape[1], inShape[2], inShape[3], 1])
OP_val  = np.reshape(OP_val  , [inShape[0], inShape[1], inShape[2], inShape[3], 1])

x_val_FWIP   = np.concatenate((x_val_FW, IP_val), axis=-1)
x_val_FWIPOP = np.concatenate((x_val_FWIP, OP_val), axis=-1)


#4.3- concatenate testing data
inShape = IP_test.shape
IP_test   = np.reshape(IP_test  , [inShape[0], inShape[1], inShape[2], inShape[3], 1])
OP_test   = np.reshape(OP_test  , [inShape[0], inShape[1], inShape[2], inShape[3], 1])

x_test_FWIP   = np.concatenate((x_test_FW, IP_test), axis=-1)
x_test_FWIPOP = np.concatenate((x_test_FWIP, OP_test), axis=-1)


# #4.4- extra lines to add more data to training data
# Data_FWIPOP = np.concatenate((Data_FWIPOP, x_val_FWIPOP), axis=0)
# Labels_tr   = np.concatenate((Labels_tr, Labels_val), axis=0)
#
# x_val_FWIPOP = x_test_FWIPOP [0:5, :, :, :, :]
# Labels_val   = Labels_test[0:5, :, :, :, :]

# extra lines to use all data as training data
Data_FWIPOP = np.concatenate((Data_FWIPOP, x_val_FWIPOP), axis=0)
Data_FWIPOP = np.concatenate((Data_FWIPOP, x_test_FWIPOP[2:10,:,:,:,:]), axis=0)
x_val_FWIPOP = x_test_FWIPOP[0:2,:,:,:,:]

Labels_tr = np.concatenate((Labels_tr, Labels_val), axis=0)
Labels_tr = np.concatenate((Labels_tr, Labels_test[2:10,:,:,:,:]), axis=0)
Labels_val = Labels_test[0:2,:,:,:,:]

print("Size of training images before the model= " + str(Data_FWIPOP.shape))
print("Size of Validation images before the model= " + str(x_val_FWIPOP.shape))
print("Size of training Labels before the model= " + str(Labels_tr.shape))
print("Size of Validation Labels before the model= " + str(Labels_val.shape))

print("All data have been reashaped for model_FatWaterIPOP")

UsedLoss =  "U_net_Generalized_Weighted_Dice_38Train_2Valid_0Test_3rd"


# print("Run Experiment: Filter_Size= " + str(Filter_Size) + ", Stride_Size= " + str(Stride_Size))
print("Run Experiment Lossfunction: " + UsedLoss)
FOV_info = UsedLoss


#4.4- build the model_FatWaterIPOP

Path_To_Weights = path_to_saved_model+'Model_'+FOV_info+'.hdf5'

# model_FWIPOP = unet(pretrained_weights = Path_To_Weights, No_Channels = 4, No_Classes = No_Of_Classes, filter_Size = 5, strides_size = 2, BottleNeck = 26)
model_FWIPOP = unet(No_Channels = 4, No_Classes = No_Of_Classes, filter_Size = 5, strides_size = 2, BottleNeck = 26)
modelFWIPOP_checkpoint = ModelCheckpoint(path_to_Weights+'Model_'+FOV_info+'.hdf5', monitor='loss', period = 5, save_best_only=False) #verbose=1  #path_to_saved_model
tb = TensorBoard(log_dir=path_to_saved_model+'logs_Reproducible_Unet/Model_9_'+FOV_info, histogram_freq=10, write_graph=True, write_images=False)

print("Satrt Training Model_FatWaterIPOP")
model_FWIPOP.fit(x = Data_FWIPOP , y = Labels_tr , validation_data = (x_val_FWIPOP, Labels_val),batch_size = BatchSize, epochs=NoEpochs,callbacks=[modelFWIPOP_checkpoint,tb], shuffle=False) #steps_per_epoch=300


print("Start testing Model")

results = model_FWIPOP.predict(Data_FWIPOP[0:3,:,:,:,:])
saveEyeResult(path_to_test+"Model_Train"+FOV_info+".npy",results)

results = model_FWIPOP.predict(x_val_FWIPOP)
saveEyeResult(path_to_test+"Model_Validation"+FOV_info+".npy",results)

results = model_FWIPOP.predict(x_test_FWIPOP)#x_test_FWIPOP[5:10,:,:,:,:]
saveEyeResult(path_to_test+"Model_Test"+FOV_info+".npy",results)

saveEyeResult(path_to_test+"Train_Labels.npy",Labels_tr[0:3,:,:,:,:])
saveEyeResult(path_to_test+"Validation_Labels.npy",Labels_val)
saveEyeResult(path_to_test+"Test_Labels.npy",Labels_test) #Labels_test[5:10,:,:,:,:]

print("Done Model_FatWater ...")

#---------------------------------------------------
