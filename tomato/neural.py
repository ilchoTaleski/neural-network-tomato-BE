import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy
import cv2
import os
from os import listdir
from os.path import isfile, join
from keras.models import model_from_json
import tensorflow as tf
from backend import urls
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from PIL import Image

def predictImage(imageFile):
    tf.keras.backend.clear_session()
    imcv = numpy.asarray(imageFile)
    img_small = cv2.resize(imcv, (128,128))
    img_small_without_alpha = img_small[:,:,:3]
    dirspot = os.getcwd()
    print(keras.__version__)

    print(img_small_without_alpha.shape)

    model = urls.get_mobile_model()
    print(model)

    with keras.backend.get_session().graph.as_default():
        prediction = model.predict(numpy.array([img_small_without_alpha]))
        return prediction

    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape = (64, 64, 3)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(2, activation='softmax'))
    # model.load_weights('tomato/assets/model4.h5')

    # json_file = open('tomato/assets/model_new.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("tomato/assets/model_glup.h5")

    # with keras.backend.get_session().graph.as_default():
    #     # model = load_model("tomato/assets/model-128-93.h5")
    #     # print("Loaded model from disk")
    #     # prediction = model.predict(numpy.array([img_small_without_alpha]))
    #     # return prediction
    #
    #     json_file = open('tomato/assets/model_224_98.json', 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     loaded_model = model_from_json(loaded_model_json)
    #     # load weights into new model
    #     loaded_model.load_weights("tomato/assets/model_224_98.h5")
    #     prediction = loaded_model.predict(numpy.array([img_small_without_alpha]))
    #     return prediction
        # return [[0,1]]
    # model = load_model('tomato/assets/model-128-93.h5')

    # model = Sequential()

    # model = keras.models.load_weights('tomato/assets/model4.h5')

def predictImages(imageFiles):
    tf.keras.backend.clear_session()

    numpyImages = []

    for x in imageFiles:
        imcv = numpy.asarray(x)
        img_small = cv2.resize(imcv, (224,224))

        img_small_without_alpha = img_small[:,:,:3]


        # img = preprocess_input(img_small_without_alpha)
        numpyImages.append(img_small_without_alpha)

    model = urls.get_model()

    with keras.backend.get_session().graph.as_default():
        prediction = model.predict(numpy.array(numpyImages))
        return prediction

def getFilters(image, layer=0):
    tf.keras.backend.clear_session()
    imcv = numpy.asarray(image)
    img_small = cv2.resize(imcv, (224, 224))
    img_small_without_alpha = img_small[:, :, :3]

    model = urls.get_model()

    count_layer = 0
    actual_layer = 0

    for x in model.layers:
        if 'conv' in x.name:
            if(count_layer == layer):
                break
            count_layer += 1
        actual_layer += 1

    model = Model(inputs=model.inputs, outputs=model.layers[actual_layer].output)

    with keras.backend.get_session().graph.as_default():

        img = preprocess_input(img_small_without_alpha)

        feature_maps = model.predict(numpy.array([img]))

        x = 1
        num_filters = len(feature_maps[0][0][0])
        filterImages = []
        for fmap in feature_maps:
            for _ in range(num_filters):
                fm_image = feature_maps[0, :, :, x-1]
                im = Image.fromarray(fm_image)
                im = im.convert('RGB')
                filterImages.append(im)
                x += 1

        return filterImages

    # img = load_img(‘/content/drive/My Drive/Colab Notebooks/random images/7262-drought (1).jpg’, target_size=(128,
    # 128))
    # img = cv2.imread(‘ / content / drive / My
    # Drive / Colab
    # Notebooks / random
    # images / 7262 - drought(1).jpg’)
    # # convert the image to an array
    # img = cv2.resize(np.array(img), (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    # # img = img_to_array(img)
    # # expand dimensions so that it represents a single ‘sample’
    # # img = expand_dims(img, axis=0)
    # # plt.imshow(img[0])
    # # prepare the image (e.g. scale pixel values for the vgg)
    # # img = preprocess_input(img)
    # plt.imshow(img)
    # plt.show()
    # # get feature map for first hidden layer
    # feature_maps = model.predict(np.array([img]))
    # # plot all 64 maps in an 8x8 squares
    # square = 8
    # ix = 1
    # plt.figure(1)
    # for fmap in feature_maps:
    #     for _ in range(2):
    #         for _ in range(16):
    #             # specify subplot and turn of axis
    #             #       ax = pyplot.subplot(12, 8, ix)
    #             #       ax.set_xticks([])
    #             #       ax.set_yticks([])
    #             #         plot filter channel in grayscale
    #             plt.imshow(feature_maps[0, :, :, ix - 1], cmap=“gray”)
    #             ix += 1
    #             plt.show()
    #     # show the figure


