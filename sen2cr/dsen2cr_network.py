import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras import Model, Input
from tensorflow.compat.v1.keras.layers import Conv2D, Concatenate, Activation, Lambda, Add, Layer

tf.disable_v2_behavior()
K.set_image_data_format('channels_first')


def resBlock(input_l, feature_size, kernel_size, scale=0.1):
    """Definition of Residual Block to be repeated in body of network."""
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(input_l)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)

    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([input_l, tmp])


def DSen2CR_model(input_shape,
                  batch_per_gpu=2,
                  num_layers=32,
                  feature_size=256,
                  use_cloud_mask=True,
                  include_sar_input=True):
    """Definition of network structure. """

    global shape_n

    # define dimensions
    input_opt = Input(shape=input_shape[0])
    input_sar = Input(shape=input_shape[1])

    if include_sar_input:
        x = Concatenate(axis=1)([input_opt, input_sar])
    else:
        x = input_opt

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    x = Activation('relu')(x)

    # main body of network as succession of resblocks
    for i in range(num_layers):
        x = resBlock(x, feature_size, kernel_size=[3, 3])

    # One more convolution
    x = Conv2D(input_shape[0][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)

    # Add first layer (long skip connection)
    x = Add()([x, input_opt])

    if use_cloud_mask:
        class AddCloudMaskLayer(Layer):
            def __init__(self, optical_input_layer_shape, **kwargs):
                super().__init__(**kwargs)
                self.shape = optical_input_layer_shape

            def call(self, inputs):
                return K.concatenate([
                    inputs,
                    K.zeros(
                        shape=(int(batch_per_gpu), 1, self.shape[2], self.shape[3])
                    )
                ], axis=1)

        x = Concatenate(axis=1)([x, input_opt])

        x = AddCloudMaskLayer(tf.shape(input_opt))(x)

    model = Model(inputs=[input_opt, input_sar], outputs=x)

    return model, shape_n
