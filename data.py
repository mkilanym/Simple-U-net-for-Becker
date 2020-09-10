from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import keras as K
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import SimpleITK

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def loadSinglenpyData(path_to_data):

    Folders = ["Fat","Water","FatFraction","IP", "OP"]
    for i,name in enumerate(Folders):
        # 1- load images
        temp_Data     =np.load(path_to_data + "Images/" + name + "/" + name + "_Images.npy") #, allow_pickle = True
        temp_Aug_Data = np.load(path_to_data + "Images/" + name + "_Aug/" + name + "_Aug_Images.npy") # , allow_pickle = True

        inShape = temp_Data.shape
        temp_Data = datasetZeroPadding(temp_Data, [inShape[0], inShape[1], inShape[2], 32])
        temp_Aug_Data = datasetZeroPadding(temp_Aug_Data, [inShape[0], inShape[1], inShape[2], 32])

        temp_Concatenate = np.concatenate((temp_Data, temp_Aug_Data), axis=0)

        # 2- concatenate
        if(i == 0):
            Fat = np.concatenate((temp_Data, temp_Aug_Data), axis=0)
        elif(i == 1):
            Water = np.concatenate((temp_Data, temp_Aug_Data), axis=0)
        elif (i == 2):
            FatFraction = np.concatenate((temp_Data, temp_Aug_Data), axis=0)
        elif (i == 3):
            IP = np.concatenate((temp_Data, temp_Aug_Data), axis=0)
        elif (i == 4):
            OP = np.concatenate((temp_Data, temp_Aug_Data), axis=0)

        print("Concatenate " + name)
        print("Size of original data = ", str(temp_Data.shape))
        print("Size of augmented data = ", str(temp_Aug_Data.shape))
        print("Size of concatenated data = ", str(temp_Concatenate.shape))

    # 3- load labels
    temp_Data     = np.load(path_to_data + "Labels/Labels_Labels.npy") #allow_pickle = True
    temp_Aug_Data = np.load(path_to_data + "Labels_Aug/Labels_Aug_Labels.npy") #, allow_pickle = True

    inShape = temp_Data.shape
    temp_Data = datasetZeroPadding(temp_Data, [inShape[0], inShape[1], inShape[2], 32])
    temp_Aug_Data = datasetZeroPadding(temp_Aug_Data, [inShape[0], inShape[1], inShape[2], 32])

    temp_Labels   = np.concatenate((temp_Data, temp_Aug_Data), axis=0)

    A = K.utils.to_categorical(temp_Labels)
    inShape = A.shape
    if(inShape[-1] == 13): # it means that 0 is included as a label, which is not accepted
        Labels = np.zeros((inShape[0], inShape[1], inShape[2], inShape[3], inShape[4] - 1))
        Labels[:, :, :, :, :] = A[:, :, :, :, 1::]
        print("Corect the labels")
    else:
        Labels = np.zeros((inShape[0], inShape[1], inShape[2], inShape[3], inShape[4]))
        Labels[:, :, :, :, :] = A[:, :, :, :, :]
        print("Keep the labels")

    # # 1.1- Load fat images
    # Fat_Data     = np.load(path_to_data + "Images/Fat/Fat_Images.npy")
    # Fat_Aug_Data = np.load(path_to_data + "Images/Fat_Aug/Fat_Aug_Images.npy")
    # # 1.2- concatenate fat images
    # # inShape = np.shape(Fat_Data)
    # #Fat_Data     = np.reshape(Fat_Data, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    # #Fat_Aug_Data = np.reshape(Fat_Aug_Data, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    # Fat = np.concatenate((Fat_Data, Fat_Aug_Data), axis=0)

    return Fat, Water, FatFraction, IP, OP, Labels


def loadnpyData(path_to_train,path_to_validate,pth_to_test,Flage):

    T2_Data = np.load(path_to_train + "Images/T2/Images.npy")  # loadMRImages(path_to_train+"Images","image")
    T1_Data = np.load(path_to_train + "Images/T1/T1_Images.npy")  # loadMRImages(path_to_train+"Images","image")
    cutSclera_Label = np.load(path_to_train + "Labels/cutSclera/cutSclera_Labels.npy")  # loadMRImages(path_to_train+"Labels","label")
    Tumor_Label = np.load(path_to_train + "Labels/Tumor/Tumor_Labels.npy")

    T2_x_val = np.load(path_to_validate + "Images/T2/Images.npy")  # loadMRImages(path_to_validate+"Images","image")
    T1_x_val = np.load(path_to_validate + "Images/T1/T1_Images.npy")  # loadMRImages(path_to_validate+"Images","image")
    cutSclera_y_val = np.load(path_to_validate + "Labels/cutSclera/cutSclera_Labels.npy")  # loadMRImages(path_to_validate+"Labels","label")
    Tumor_y_val = np.load(path_to_validate + "Labels/Tumor/Tumor_Labels.npy")

    T2_x_test = np.load(pth_to_test + "Images/T2/Images.npy")  # loadMRImages(pth_to_test+"Images","image")
    T1_x_test = np.load(pth_to_test + "Images/T1/T1_Images.npy")  # loadMRImages(pth_to_test+"Images","image")



    # reshape the every array to include the channel at the end
    inShape = np.shape(T2_Data)
    T2_Data = np.reshape(T2_Data, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    T1_Data = np.reshape(T1_Data, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    Data = np.concatenate((T2_Data,T1_Data),axis = -1)

    cutSclera_Label = np.reshape(cutSclera_Label, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    Tumor_Label = np.reshape(Tumor_Label, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    Label = np.concatenate((cutSclera_Label, Tumor_Label), axis=-1)


    inShape = np.shape(T2_x_val)
    T2_x_val = np.reshape(T2_x_val, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    T1_x_val = np.reshape(T1_x_val, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    x_val = np.concatenate((T2_x_val,T1_x_val),axis = -1)

    cutSclera_y_val = np.reshape(cutSclera_y_val, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    Tumor_y_val = np.reshape(Tumor_y_val, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    y_val = np.concatenate((cutSclera_y_val, Tumor_y_val), axis=-1)

    inShape = np.shape(T2_x_test)
    T2_x_test = np.reshape(T2_x_test, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    T1_x_test = np.reshape(T1_x_test, (inShape[0], inShape[1], inShape[2], inShape[3], 1))
    x_test = np.concatenate((T2_x_test, T1_x_test), axis=-1)

    return Data, Label, x_val, y_val, x_test

def loadMRImages(Path,MRDataType):
    Load_Image = []
    for file in os.listdir(Path):
        inputImage = SimpleITK.ReadImage(Path + "/" + file)
        numpyImage = SimpleITK.GetArrayFromImage(inputImage) # out z,y,x
        numpyImage = np.transpose(numpyImage,(-1,1,0))
        #print(np.shape(numpyImage))

        if(MRDataType == "image"):
            inputImage = normalizeData(numpyImage)
        elif(MRDataType == "label"):
            inputImage = checkLabel(numpyImage)
        Load_Image.append(numpyImage)
    return np.array(Load_Image)

def normalizeData(img):
    max_intensity = np.amax(img)
    img = img/max_intensity
    return img


def checkLabel(Label):
    InShape = Label.shape
    Out = np.zeros((InShape[0],InShape[1],InShape[2]))
    Out[:,:,:]= Label>0
    return Out

def datasetZeroPadding(Data,TargetSahpe):
    inShape = Data.shape
    Flag = 0
    if(len(inShape) == len(TargetSahpe)):
        for i in range(0,len(inShape)):
            if(inShape[i] <= TargetSahpe[i]):
                Flag += 1

        if(Flag == len(inShape)):
            Out = np.zeros((TargetSahpe))
            if(len(inShape) == 4):
                Out[(TargetSahpe[0]-inShape[0])//2:(TargetSahpe[0]+inShape[0])//2,(TargetSahpe[1]-inShape[1])//2:(TargetSahpe[1]+inShape[1])//2,(TargetSahpe[2]-inShape[2])//2:(TargetSahpe[2]+inShape[2])//2,(TargetSahpe[3]-inShape[3])//2:((TargetSahpe[3]+inShape[3])//2)] = Data
            elif (len(inShape) == 3):
                Out[(TargetSahpe[0] - inShape[0])/2: (TargetSahpe[0] + inShape[0])/2,
                (TargetSahpe[1] - inShape[1])/2: (TargetSahpe[1] + inShape[1])/2,
                (TargetSahpe[2] - inShape[2])/2: (TargetSahpe[2] + inShape[2])/2] = Data
            return Out
        else:
            return Data
    else:
        return Data



  
def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def custometestGenerator(test_path,no_channels = 2):
    file_list = os.listdir(test_path + 'Labels/')
    test_path_newe = test_path + 'Images'
	
    for i, ID in enumerate(file_list):
	
        Fat = np.load(test_path_newe + '/Fat/' + ID)
        dim = Fat.shape
        Fat = np.reshape(Fat  , [dim[0], dim[1], dim[2], 1])
			
        Water = np.load(test_path_newe + '/Water/' + ID)
        Water = np.reshape(Water  , [dim[0], dim[1], dim[2], 1])
			
        Fat_Water = np.concatenate((Fat, Water), axis=-1)
			
        if(no_channels == 2):
            img = Fat_Water
				
        elif(no_channels == 3):
            FatFraction = np.load(test_path_newe + '/FatFraction/' + ID)
            FatFraction = np.reshape(FatFraction  , [dim[0], dim[1], dim[2], 1])
				
            img = np.concatenate((Fat_Water, FatFraction), axis=-1)
			
        elif(no_channels == 4):
            IP = np.load(test_path_newe + '/IP/' + ID)
            IP = np.reshape(IP  , [dim[0], dim[1], dim[2], 1])
				
            OP = np.load(test_path_newe + '/OP/' + ID)
            OP = np.reshape(OP  , [dim[0], dim[1], dim[2], 1])
				
            Fat_Water_IP = np.concatenate((Fat_Water, IP), axis=-1)
            img = np.concatenate((Fat_Water_IP, OP), axis=-1)
				
        elif(no_channels == 5):
            FatFraction = np.load(test_path_newe + '/FatFraction/' + ID)
            FatFraction = np.reshape(FatFraction  , [dim[0], dim[1], dim[2], 1])
				
            IP = np.load(test_path_newe + '/IP/' + ID)
            IP = np.reshape(IP  , [dim[0], dim[1], dim[2], 1])
				
            OP = np.load(test_path_newe + '/OP/' + ID)
            OP = np.reshape(OP  , [dim[0], dim[1], dim[2], 1])
				
            Fat_Water_Fraction = np.concatenate((Fat_Water, FatFraction), axis=-1)
            Fat_Water_Fraction_IP = np.concatenate((Fat_Water_Fraction, IP), axis=-1)
            img = np.concatenate((Fat_Water_Fraction_IP, OP), axis=-1)

        img = np.reshape(img, (1,) + img.shape)
        yield img
		

def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


def saveEyeResult(save_path,npyfile):
    np.save(save_path , npyfile) #np.save(save_path+"/Test_Results.npy",npyfile)
    #for i,item in enumerate(npyfile):
    #    SimpleITK.WriteImage(item,os.path.join(save_path,"%d_predict.mha"%i))