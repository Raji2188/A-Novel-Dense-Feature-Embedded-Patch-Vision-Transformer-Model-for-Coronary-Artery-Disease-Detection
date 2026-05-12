import numpy as np
import pandas as pd
import pydicom
import cv2
import os
from Preprocess import *
from sklearn.model_selection import train_test_split
from Classification import *
from keras.models import load_model



def Train():
    
    fold_dir="Non-contrast Cardiac CT Images Dataset with Coronary Artery Calcium Scoring/Dataset"
    folder_dir_list=os.listdir(fold_dir)
    
    for dirs in folder_dir_list:
        fl_list=fold_dir+"/"+dirs
        for j in os.listdir(fl_list):
           
            k=fl_list+"/"+j
            for img_dir in os.listdir(k):
                if not img_dir.endswith('.png'):
                     imgs_path=k+"/"+img_dir
                     cmt=0
                     for img_names in os.listdir(imgs_path):
                         image_path=imgs_path+"/"+img_names
                         dicom_data = pydicom.dcmread(image_path)
                         image_data_ = dicom_data.pixel_array
                         image_data = cv2.normalize(image_data_, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                         image_data=cv2.resize(image_data,(256,256))
                         preprocessed_imge=GABF(image_data)
                         if dirs=="Healthy":
                              new_dir_path = "Healthy_images/"+j
                              if not os.path.exists(new_dir_path):
                                  # If it doesn't exist, create it
                                  os.makedirs(new_dir_path)
                                  print(f"Directory '{new_dir_path}' created.")
                              else:
                                  print(f"Directory '{new_dir_path}' already exists.")
                             
                         else:
                            class_name="coronary artery disease"
                            new_dir_path = "Coronery/"+j
                            if not os.path.exists(new_dir_path):
                                # If it doesn't exist, create it
                                os.makedirs(new_dir_path)
                                print(f"Directory '{new_dir_path}' created.")
                            else:
                                print(f"Directory '{new_dir_path}' already exists.")
                            cv2.imwrite(new_dir_path+"/"+img_names+".png",image_data)
                         
    
    original_images,preprocessed_images,labels=[],[],[]
    image_fold="Dataset"
    image_fold_list=os.listdir(image_fold)
    for image_folders in image_fold_list:
        if image_folders=="Healthy_images":
            img_sub_dir_path=image_fold+"/"+image_folders
            img_sub_dir_list=os.listdir(img_sub_dir_path)
            for image_Sub_dir in img_sub_dir_list:
                images_path=img_sub_dir_path+"/"+image_Sub_dir
                images_path_list=os.listdir(images_path)
                for images in images_path_list:
                    image_path=images_path+"/"+images
                    image=cv2.imread(image_path)
                    original_images.append(image)
                    "----------------------------Preprocess--------------------------"
                    preprocessed_image=GABF(image)
                    preprocessed_images.append(preprocessed_image)
                    labels.append(image_folders.split("_")[0])
        elif image_folders=="Coronery":
            img_sub_dir_path=image_fold+"/"+image_folders
            img_sub_dir_list=os.listdir(img_sub_dir_path)
            for image_Sub_dir in img_sub_dir_list:
                images_path=img_sub_dir_path+"/"+image_Sub_dir
                images_path_list=os.listdir(images_path)
                for images in images_path_list:
                    image_path=images_path+"/"+images
                    image=cv2.imread(image_path)
                    original_images.append(image)
                    preprocessed_image=GABF(image)
                    preprocessed_images.append(preprocessed_image)
                    labels.append("Coronary Artery Disease")
            
    
    # np.save("Features/Original_Images.npy",original_images)
    # np.save("Features/Preprocessed_Imges.npy",preprocessed_images)
    # np.save("Features/Labels.npy",labels)
    
    
    
    "-------------------------Classification---------------------------------"
    # densely connected embedded patch vision transformer model 
    preprocessed_images=np.load("Features/Preprocessed_Imges.npy")  
    labels=np.load("Features/Labels.npy")
    labels[labels=='Healthy']=0;labels[labels=='Coronary Artery Disease']=1
    labels=labels.astype(int)
    x_train,x_test,y_train,y_test=train_test_split(preprocessed_images,labels,test_size=0.2,random_state=42)
    
    # np.save("Features/x_test.npy",x_test)
    # np.save("Features/y_test.npy",y_test)
    dcepvtm=DCEPVTM(x_train_,y_train)
    dcepvtm.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    # dcepvtm.save("Model/Proposed_model.h5")
    