import glob
import random

from PIL import Image
import numpy as np

def load_Images():
    image1_pattern = r"D:\change_detection_dataset\time1\*.png"
    image1_list = glob.glob(image1_pattern)
    random.shuffle(image1_list)

    training_rate = 0.1
    training_num = round(len(image1_list) * training_rate)

    training_list = image1_list[0:training_num]
    testing_list = image1_list[training_num:]

    Train_Image_dataset = np.array([])
    Train_Label_dataset = np.array([])
    for training_image1_path in training_list:
        image1_tmp = np.array(Image.open(training_image1_path)).astype("float32")/255
        image2_tmp = np.array(Image.open(training_image1_path.replace("time1", "time2"))).astype("float32")/255
        label_tmp = np.array(Image.open(training_image1_path.replace("time1", "label"))).astype("float32")/255

        image_tmp = np.array([image1_tmp, image2_tmp])
        Train_Image_dataset = np.append(Train_Image_dataset, image_tmp)
        Train_Label_dataset = np.append(Train_Label_dataset, label_tmp)

    Train_Image_dataset = Train_Image_dataset.reshape([-1, 2, 256, 256, 3])
    Train_Label_dataset = Train_Label_dataset.reshape([-1, 256, 256, 1])

    # Test_Image_dataset = np.array([])
    # Test_Label_dataset = np.array([])
    # for testing_image1_path in testing_list:
    #     image1_tmp = np.array(Image.open(testing_image1_path))
    #     image2_tmp = np.array(Image.open(testing_image1_path.replace("time1", "time2")))
    #     label_tmp = np.array(Image.open(testing_image1_path.replace("time1", "label")))
    #
    #     image_tmp = np.array([image1_tmp, image2_tmp])
    #     Test_Image_dataset = np.append(Test_Image_dataset, image_tmp)
    #     Test_Label_dataset = np.append(Test_Label_dataset, label_tmp)
    #
    # Test_Image_dataset = Test_Image_dataset.reshape([-1, 2, 256, 256, 3])
    # Test_Label_dataset = Test_Label_dataset.reshape([-1, 256, 256, 1])

    # return Train_Image_dataset,Train_Label_dataset,Test_Image_dataset,Test_Label_dataset
    return Train_Image_dataset,Train_Label_dataset

load_Images()
