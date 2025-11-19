from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization
from tensorflow.keras.models import Model

def UNet(inputs, activation):
    # Encoder
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, (3,3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(512, (3,3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Bottleneck
    bottleneck = Conv2D(1024, (3,3), activation="relu", padding="same")(pool4)
    bottleneck = Conv2D(1024, (3,3), activation="relu", padding="same")(bottleneck)

    # Decoder
    upConv1 = Conv2DTranspose(512, (2,2), strides=2, padding="same")(bottleneck)
    concat1 = concatenate([upConv1, conv4])
    conv5 = Conv2D(512, (3,3), activation="relu", padding="same")(concat1)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    upConv2 = Conv2DTranspose(256, (2, 2), strides=2, padding="same")(conv5)
    concat2 = concatenate([upConv2, conv3])
    conv6 = Conv2D(256, (3,3), activation="relu", padding="same")(concat2)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    upConv3 = Conv2DTranspose(128, (2, 2), strides=2, padding="same")(conv6)
    concat3 = concatenate([upConv3, conv2])
    conv7 = Conv2D(128, (3,3), activation="relu", padding="same")(concat3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    upConv4 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(conv7)
    concat4 = concatenate([upConv4, conv1])
    conv8 = Conv2D(64, (3,3), activation="relu", padding="same")(concat4)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    # Output: 2 channels (Disc + Cup) with sigmoid
    outputs = Conv2D(1, (1,1), activation=activation)(conv8)

    model = Model(inputs, outputs)

    return model