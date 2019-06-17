from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np


def generate_batch(seed, batch_size, image1_path, image2_path, label_path):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    images1_generator = datagen.flow_from_directory(
        image1_path,
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode=None,
        # save_to_dir='E:/tmp/augment4',
        save_format='png',
        seed=seed,
        shuffle=True)
    images2_generator = datagen.flow_from_directory(
        image2_path,
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode=None,
        # save_to_dir='E:/tmp/augment4',
        save_format='png',
        seed=seed,
        shuffle=True)
    label_generator = datagen.flow_from_directory(
        label_path,
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode=None,
        # save_to_dir='E:/tmp/augment5',
        save_format='png',
        seed=seed,
        shuffle=True)

    image_generator = map(lambda x, y: np.append(x[:, np.newaxis, :, :], y[:, np.newaxis, :, :], axis=1),
                          images1_generator, images2_generator)
    input_batch = zip(image_generator,label_generator)
    return input_batch

#     'D:/change_detection_dataset/time1',

#     'D:/change_detection_dataset/time2',

#     'D:/change_detection_dataset/label',
# train_generator = generate_batch(1,1,'D:/change_detection_dataset/time1','D:/change_detection_dataset/time2','D:/change_detection_dataset/label')
#
# for t in train_generator:
#     print(1)

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=2000,
#     epochs=50)
