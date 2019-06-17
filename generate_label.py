from PIL import Image
import random
import glob
import os
import math
from multiprocessing import Pool

image1_pattern = r"D:\SZTAKI_AirChange_Benchmark\*\*\im1.bmp"
image1_list = glob.glob(image1_pattern)

size = 256
threshold = 0.3

image1_output = r"D:\change_detection_dataset\time1"
image2_output = r"D:\change_detection_dataset\time2"
label_output = r"D:\change_detection_dataset\label"

count = 0
for num, image1_path in enumerate(image1_list):
    image2_path = image1_path.replace("im1","im2")
    label_path = image1_path.replace("im1","gt")


    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    label = Image.open(label_path)

    X = image1.width
    Y = image1.height



    for xmax in range(size, X, int(size * threshold)):
        xmin = xmax - size
        for ymax in range(size, Y, int(size * threshold)):
            ymin = ymax - size

            region = (xmin, ymin, xmax, ymax)

            random_shift_X = random.choice([-2, -1, -1, 1, 1, 2])
            random_shift_Y = random.choice([-2, -1, -1, 1, 1, 2])
            xmin_s = xmin + random_shift_X if xmin + random_shift_X >= 0 and xmax + random_shift_X <= X else xmin
            xmax_s = xmax + random_shift_X if xmin + random_shift_X >= 0 and xmax + random_shift_X <= X else xmax
            ymin_s = ymin + random_shift_Y if ymin + random_shift_Y >= 0 and ymax + random_shift_Y <= Y else ymin
            ymax_s = ymax + random_shift_Y if ymin + random_shift_Y >= 0 and ymax + random_shift_Y <= Y else ymax

            region_s = (xmin_s, ymin_s, xmax_s, ymax_s)

            image_name = os.path.join(str(num) + "_" + str(xmin) + "_" + str(ymin) + ".png")
            print(image_name)
            count = count+1

            # cropImg1 = image1.crop(region)
            cropImg1 = image1.crop(region_s)
            image1_output_fullpath = os.path.join(image1_output, image_name)
            cropImg1.save(image1_output_fullpath)

            cropImg2 = image2.crop(region)
            image2_output_fullpath = os.path.join(image2_output,image_name)
            cropImg2.save(image2_output_fullpath)

            cropLabel = label.crop(region)
            label_output_fullpath = os.path.join(label_output, image_name)
            cropLabel.save(label_output_fullpath)

print(count)