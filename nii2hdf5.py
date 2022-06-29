
# Author: S. M. Kamrul Hasan (sh3190@rit.edu)
# h5 conversion from nifty files
#slice seperation
import numpy as np
import nibabel as nb
import glob
import matplotlib.pyplot as plt
import cv2
import h5py
import os
from tqdm import tqdm
## the directory with datatset
root_dir = "/mnt/media/cardiac_main/ACDC_100_training/*/*"

files = glob.glob(root_dir)
labels = []
images = []
test_list = ['001','007','008','011','013','022','024','033','052','059','064','065','066','068','075','080','081','083','084','093']
for each in files:
    if "frame" in each and "gt" in each:
        labels.append(each)
    elif "frame" in each:
        images.append(each)

os.makedirs('/mnt/media/cardiac_main/data/ACDC_test/data/slices',exist_ok=True)
prev_patient = "patient001"
slice_num = 1
train_file = open('/mnt/media/cardiac_main/data/ACDC_test/train_slices.list','w')
test_file = open('/mnt/media/cardiac_main/data/ACDC_test/test_slices.list','w')
test_h5_file = open('/mnt/media/cardiac_main/data/ACDC_test/all_slices.list','w')
for i in tqdm(range(len(images))):
    slice_num=1
    patient = images[i].split("/")[-2]
    image = nb.load(images[i]).get_data()
    label = nb.load(labels[i]).get_data()
    image = image/255
    size = image.shape
    if size[0]>size[1]:
        pad_left = (size[0]-size[1])//2+(size[0]-size[1])%2
        pad_right = (size[0]-size[1])//2
        image = np.pad(image,((0,0),(pad_left,pad_right),(0,0)),'constant')
        label = np.pad(label,((0,0),(pad_left,pad_right),(0,0)),'constant')
    elif size[0]<size[1]:
        pad_up = (size[1]-size[0])//2+(size[1]-size[0])%2
        pad_down = (size[1]-size[0])//2
        image = np.pad(image,((pad_up,pad_down),(0,0),(0,0)),'constant')
        label = np.pad(label,((pad_up,pad_down),(0,0),(0,0)),'constant')
        
    assert image.shape[0]==image.shape[1],'padding failed'
    assert image.shape[2]==label.shape[2],f'{image.shape[2],label.shape[2],images[i],labels[i]}'
    slices = image.shape[2]

    if i!=0 and prev_patient == patient:
        slice_num = slice_num +slices
    frames = [1,2]

    if patient[-3:] not in test_list:
        for num in range(slices):
            case_image = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_NEAREST)
            case_label = cv2.resize(label[:,:,num],(224,224),interpolation=cv2.INTER_NEAREST)  
            
            if slice_num % 2 == 0:
                hp = h5py.File("/mnt/media/cardiac_main/data/ACDC_test/data/slices/" + str(patient) + "_frame01"  + "_slice_" + str(int(slice_num/2)) + '.h5', 'w')
                hp.create_dataset('image', data=case_image)
                hp.create_dataset('label', data=case_label)
                train_file.write(str(patient) + "_frame01" + "_slice_" + str(int(slice_num/2)) +'\n')
                slice_num+=1

            else:

                hp = h5py.File("/mnt/media/cardiac_main/data/ACDC_test/data/slices/" + str(patient) + "_frame02"  + "_slice_" + str(int(slice_num/2+1))  + '.h5', 'w')
                hp.create_dataset('image', data=case_image)
                hp.create_dataset('label', data=case_label)
                train_file.write(str(patient) + "_frame02" + "_slice_" + str(int(slice_num/2+1)) +'\n')
                slice_num+=1


    else:
        image_h5, label_h5 = np.zeros((slices, 224, 224)), np.zeros((slices, 224, 224))
        for num in range(slices):
            case_image = cv2.resize(image[:,:,num],(224,224),interpolation=cv2.INTER_NEAREST)
            case_label = cv2.resize(label[:,:,num],(224,224),interpolation=cv2.INTER_NEAREST)
            image_h5[num], label_h5[num] = case_image, case_label
            if slice_num % 2 == 0:
                hp = h5py.File("/mnt/media/cardiac_main/data/ACDC_test/data/slices/" + str(patient) + "_frame01" + "_slice_" + str(int(slice_num/2))  + '.h5', 'w')
                hp.create_dataset('image', data=case_image)
                hp.create_dataset('label', data=case_label)
                test_file.write(str(patient) + "_frame01" + "_slice_" + str(int(slice_num/2))  +'\n')
                slice_num+=1
            else:
                hp = h5py.File("/mnt/media/cardiac_main/data/ACDC_test/data/slices/" + str(patient) + "_frame02" + "_slice_" + str(int(slice_num/2+1))  + '.h5', 'w')
                hp.create_dataset('image', data=case_image)
                hp.create_dataset('label', data=case_label)
                test_file.write(str(patient) + "_frame02" + "_slice_" + str(int(slice_num/2+1))  +'\n')
                slice_num+=1
            
    prev_patient = patient
train_file.close()
test_file.close()















# # h5 conversion from nifty files
# # frame separation
# import numpy as np
# import nibabel as nb
# import glob
# import matplotlib.pyplot as plt
# import cv2
# import h5py
# import os
# from tqdm import tqdm
# ## the directory with datatset
# root_dir = "/mnt/media/cardiac_main/ACDC_100_training/*/*"

# files = glob.glob(root_dir)
# labels = []
# images = []
# test_list = ['001','007','008','011','013','022','024','033','052','059','064','065','066','068','075','080','081','083','084','093']
# for each in files:
#     if "frame" in each and "gt" in each:
#         labels.append(each)
#     elif "frame" in each:
#         images.append(each)
# os.makedirs('/mnt/media/cardiac_main/data/ACDC_test/data',exist_ok=True)
# prev_patient = "patient001"
# train_file = open('/mnt/media/cardiac_main/data/ACDC_test/train.list','w')
# test_file = open('/mnt/media/cardiac_main/data/ACDC_test/test.list','w')
# for i in tqdm(range(len(images))):
#     patient = images[i].split("/")[-2]
#     image = nb.load(images[i]).get_data()
#     label = nb.load(labels[i]).get_data()
#     image = image/255
#     size = image.shape
#     if size[0]>size[1]:
#         pad_left = (size[0]-size[1])//2+(size[0]-size[1])%2
#         pad_right = (size[0]-size[1])//2
#         image = np.pad(image,((0,0),(pad_left,pad_right),(0,0)),'constant')
#         label = np.pad(label,((0,0),(pad_left,pad_right),(0,0)),'constant')
#     elif size[0]<size[1]:
#         pad_up = (size[1]-size[0])//2+(size[1]-size[0])%2
#         pad_down = (size[1]-size[0])//2
#         image = np.pad(image,((pad_up,pad_down),(0,0),(0,0)),'constant')
#         label = np.pad(label,((pad_up,pad_down),(0,0),(0,0)),'constant')
        
#     assert image.shape[0]==image.shape[1],'padding failed'
#     assert image.shape[2]==label.shape[2],f'{image.shape[2],label.shape[2],images[i],labels[i]}'

#     if patient[-3:] not in test_list:
#         case_image = cv2.resize(image,(224,224),interpolation=cv2.INTER_NEAREST)
#         case_label = cv2.resize(label,(224,224),interpolation=cv2.INTER_NEAREST)  
        
#         hp = h5py.File("/mnt/media/cardiac_main/data/ACDC_test/data/" + str(patient) + "_frame01" + '.h5', 'w')
#         hp.create_dataset('image', data=case_image)
#         hp.create_dataset('label', data=case_label)
#         train_file.write(str(patient) + "_frame01" +'\n')

#         hp = h5py.File("/mnt/media/cardiac_main/data/ACDC_test/data/" + str(patient) + "_frame02" + '.h5', 'w')
#         hp.create_dataset('image', data=case_image)
#         hp.create_dataset('label', data=case_label)
#         train_file.write(str(patient) + "_frame02" +'\n')


#     else:
#         case_image = cv2.resize(image,(224,224),interpolation=cv2.INTER_NEAREST)
#         case_label = cv2.resize(label,(224,224),interpolation=cv2.INTER_NEAREST)

#         hp = h5py.File("/mnt/media/cardiac_main/data/ACDC_test/data/" + str(patient) + "_frame01" + '.h5', 'w')
#         hp.create_dataset('image', data=case_image)
#         hp.create_dataset('label', data=case_label)
#         test_file.write(str(patient) + "_frame01" +'\n')

#         hp = h5py.File("/mnt/media/cardiac_main/data/ACDC_test/data/" + str(patient) + "_frame02" + '.h5', 'w')
#         hp.create_dataset('image', data=case_image)
#         hp.create_dataset('label', data=case_label)
#         test_file.write(str(patient) + "_frame02"  +'\n')
            
#     prev_patient = patient
# train_file.close()
# test_file.close()
