import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Input, GaussianNoise, Conv2D, BatchNormalization, Activation, MaxPooling2D, \
    Lambda, Add, Dropout, Flatten
from tensorflow import pad
from tensorflow.nn import pool


def model_for_training(input_shape=(63, 63, 7), out_markers_num=5):
    inputs = Input(shape=input_shape)

    # Adding Gaussian noise for training
    a = GaussianNoise(stddev=0.1)(inputs)

    x = Conv2D(64, kernel_size=(4, 4), strides=(1, 1), name='conv1')(a)
    x = BatchNormalization(name='batch1')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), name='conv2')(x)
    x = BatchNormalization(name='batch2')(x)

    a = Conv2D(64, kernel_size=(6, 6), name="addconv1")(a)
    x = Add()([x, a])

    x = Activation('relu')(x)
    a = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name='max1')(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=2, name='conv3')(a)
    x = BatchNormalization(name='batch3')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=2, name='conv4')(x)
    x = BatchNormalization(name='batch4')(x)

    a = Conv2D(128, kernel_size=(5, 5), dilation_rate=2, name="addconv2")(a)
    x = Add()([x, a])

    x = Activation('relu')(x)
    a = Lambda(lambda x: pool(x, (2, 2), 'MAX', padding='VALID', dilation_rate=(2, 2)), name='max2')(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=4, name='conv5')(a)
    x = BatchNormalization(name='batch5')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=4, name='conv6')(x)
    x = BatchNormalization(name='batch6')(x)

    a = Conv2D(256, kernel_size=(5, 5), dilation_rate=4, name="addconv3")(a)
    x = Add()([x, a])

    x = Activation('relu')(x)
    x = Lambda(lambda x: pool(x, (2, 2), 'MAX', padding='VALID', dilation_rate=(4, 4)), name='max3')(x)

    x = Conv2D(512, kernel_size=(4, 4), strides=(1, 1), dilation_rate=(8, 8), name='conv7')(x)
    x = BatchNormalization(name='batch7')(x)
    x = Activation('relu')(x)

    x = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), name='conv8')(x)
    dist_intermediate = Activation('relu')(x)

    # final branch to predict distance transformation
    x = Dropout(0.2)(dist_intermediate)
    x = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), name='conv9')(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation="linear", name='conv10')(x)
    x = Flatten(name='distance_output')(x)

    # final branch to predict cell phenotype
    u = Dropout(0.2)(dist_intermediate)

    u = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), name='conv15')(u)  # 1 -> 1
    u = Activation('relu')(u)
    u = Dropout(0.2)(u)

    u = Conv2D(out_markers_num, kernel_size=(1, 1), strides=(1, 1), activation="linear", name='conv16')(u)
    u = Flatten(name='phenotype_output')(u)

    M = tf.keras.models.Model(
        inputs=inputs,
        outputs=[x, u])

    return M


def model_for_inference(input_shape=(None, None, 7), weights=None, out_markers_num=5, window=30):
    inputs = Input(shape=input_shape)

    a = Lambda(lambda x: pad(x, [[0, 0], [window, window], [window, window], [0, 0]], "REFLECT"))(inputs)

    x = Conv2D(64, kernel_size=(4, 4), strides=(1, 1), name='conv1')(a)
    x = BatchNormalization(name='batch1')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), name='conv2')(x)
    x = BatchNormalization(name='batch2')(x)

    a = Conv2D(64, kernel_size=(6, 6), name="addconv1")(a)
    x = Add()([x, a])

    x = Activation('relu')(x)
    a = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name='max1')(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=2, name='conv3')(a)
    x = BatchNormalization(name='batch3')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=2, name='conv4')(x)
    x = BatchNormalization(name='batch4')(x)

    a = Conv2D(128, kernel_size=(5, 5), dilation_rate=2, name="addconv2")(a)
    x = Add()([x, a])

    x = Activation('relu')(x)
    a = Lambda(lambda x: pool(x, (2, 2), 'MAX', padding='VALID', dilation_rate=(2, 2)), name='max2')(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=4, name='conv5')(a)
    x = BatchNormalization(name='batch5')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=4, name='conv6')(x)
    x = BatchNormalization(name='batch6')(x)

    a = Conv2D(256, kernel_size=(5, 5), dilation_rate=4, name="addconv3")(a)
    x = Add()([x, a])

    x = Activation('relu')(x)
    x = Lambda(lambda x: pool(x, (2, 2), 'MAX', padding='VALID', dilation_rate=(4, 4)), name='max3')(x)

    x = Conv2D(512, kernel_size=(4, 4), strides=(1, 1), dilation_rate=(8, 8), name='conv7')(x)
    x = BatchNormalization(name='batch7')(x)
    x = Activation('relu')(x)

    x = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), name='conv8')(x)
    dist_intermediate = Activation('relu')(x)

    x = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), name='conv9')(dist_intermediate)
    x = Activation('relu')(x)

    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='linear', name='conv10')(x)


    x = Lambda(lambda x: x, name='distance_output')(x) # do this weird identity function thing to have an extra
                                                       # layer to name, in the same way as training model was named

    u = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), name='conv15')(dist_intermediate)
    u = Activation('relu')(u)

    u = Conv2D(out_markers_num, kernel_size=(1, 1), strides=(1, 1), activation='linear', name='conv16')(u)

    u = Lambda(lambda x: x, name='phenotype_output')(u)
                # do this weird identity function thing to have an extra
                # layer to name, in the same way as training model was named

    pred_model = tf.keras.models.Model(
    inputs=inputs,
    outputs=[x, u])

    if weights is not None:
        pred_model.load_weights(weights, by_name=True)

    return pred_model


def compress_model(input_name, output_name):

    input_path = "{}.hdf5".format(input_name)
    input_model = tf.keras.models.load_model(
        input_path, custom_objects={"pool": tf.nn.pool}, compile=False
    )
    in_channels_num = input_model.layers[0].input_shape[0][3]
    out_markers_num = input_model.layers[-1].input_shape[-1]

    model_for_inference(
        input_shape=(None, None, in_channels_num),
        out_markers_num=out_markers_num,
        weights=input_path,
    ).save("{}.h5".format(output_name))


def compress(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="model_cp",
        required=False,
        help="a file name of the model to compress without extension",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model",
        required=False,
        help="a file name to save the compressed model without extension",
    )

    args = parser.parse_args(argv)
    in_file_name = args.input
    out_file_name = args.output

    compress_model(in_file_name, out_file_name)


if __name__ == "__main__":
    # Very first argument determines action
    actions = {"compress": compress}

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print("Usage: " + "/".join(actions.keys()) + " ...")
    else:
        action(sys.argv[2:])
