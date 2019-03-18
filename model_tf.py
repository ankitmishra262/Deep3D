import tensorflow as tf


class UpSampleModel(object):
    def __init__(self, input_shape):
        super(UpSampleModel, self).__init__()
        self.width = input_shape[1]
        self.height = input_shape[2]
        self.c_dim = input_shape[0]

        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.width, self.height, self.c_dim], name="images")

        model = self.model()

    def model(self):
        conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                                         activation=tf.nn.relu, name='conv1_1')(self.images)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, name='pool1')(conv1_1)

        conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                         activation=tf.nn.relu, name='conv2_1')(pool1)
        pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, name='pool2')(conv2_1)

        conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                                         activation=tf.nn.relu, name='conv3_1')(pool2)
        conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                                         activation=tf.nn.relu, name='conv3_2')(conv3_1)
        pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, name='pool3')(conv3_2)

        conv4_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                                         activation=tf.nn.relu, name='conv4_1')(pool3)
        conv4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                                         activation=tf.nn.relu, name='conv4_2')(conv4_1)
        pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, name='pool4')(conv4_2)

        conv5_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                                         activation=tf.nn.relu, name='conv5_1')(pool4)
        conv5_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                                         activation=tf.nn.relu, name='conv5_2')(conv5_1)
        pool5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, name='pool5')(conv5_2)

        x = pool5.get_shape().as_list()[-1] * pool5.get_shape().as_list()[-2]
        fn_input = tf.keras.backend.reshape(x=pool5, shape=[-1, x])

        fc6 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu, name='fc6')(fn_input)
        drop6 = tf.keras.layers.Dropout(rate=0.5, name='drop6')(fc6)

        fc7 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu, name='fc7')(drop6)
        drop7 = tf.keras.layers.Dropout(rate=0.5, name='drop7')(fc7)

        fc8 = tf.keras.layers.Dense(units=33 * 12 * 5, activation=tf.nn.relu, name='fc8')(drop7)

        batch_norm_input = tf.keras.backend.reshape(x=fc8, shape=[-1, 33, 5, 12])
        batch_norm4 = tf.keras.layers.BatchNormalization(name='batch_norm4')(batch_norm_input)
        conv4 = tf.keras.layers.Conv2D(filters=33, kernel_size=[3, 3], padding='same',
                                       name='conv4')(batch_norm4)
        batch_norm3 = tf.keras.layers.BatchNormalization(name='batch_norm3')(conv4)
        conv3 = tf.keras.layers.Conv2D(filters=33, kernel_size=[3, 3], padding='same',
                                       name='conv3')(batch_norm3)
        batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm2')(conv3)
        conv2 = tf.keras.layers.Conv2D(filters=33, kernel_size=[3, 3], padding='same',
                                       name='conv2')(batch_norm2)
        batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm4')(conv2)
        conv1 = tf.keras.layers.Conv2D(filters=33, kernel_size=[3, 3], padding='same',
                                       name='conv1', activation=tf.nn.relu)(batch_norm1)

        scale = 1
        deconv1 = tf.keras.layers.Conv2DTranspose(filters=33, kernel_size=[1, 1], strides=(1, 1),
                                                  padding='valid', activation=tf.nn.relu, name='deconv1')(conv1)
        scale *= 2
        deconv2 = tf.keras.layers.Conv2DTranspose(filters=33, kernel_size=[2 * scale, 2 * scale],
                                                  strides=[scale, scale], padding='valid', activation=tf.nn.relu,
                                                  use_bias=False, name='deconv2')(deconv1)
        scale *= 2
        deconv3 = tf.keras.layers.Conv2DTranspose(filters=33, kernel_size=[2 * scale, 2 * scale],
                                                  strides=[scale, scale], padding='valid', activation=tf.nn.relu,
                                                  use_bias=False, name='deconv3')(deconv2)
        scale *= 2
        deconv4 = tf.keras.layers.Conv2DTranspose(filters=33, kernel_size=[2 * scale, 2 * scale],
                                                  strides=[scale, scale], padding='valid', activation=tf.nn.relu,
                                                  use_bias=False, name='deconv4')(deconv3)
        scale *= 2
        deconv5 = tf.keras.layers.Conv2DTranspose(filters=33, kernel_size=[2 * scale, 2 * scale],
                                                  strides=[scale, scale], padding='valid', activation=tf.nn.relu,
                                                  use_bias=False, name='deconv5')(deconv4)
        feats = tf.keras.layers.add(inputs=[deconv1, deconv2, deconv3, deconv4, deconv5])
        feats = tf.keras.layers.ReLU()(feats)
        scale = 2
        deconv6 = tf.keras.layers.Conv2DTranspose(filters=33, kernel_size=[2 * scale, 2 * scale],
                                                  strides=[scale, scale], padding='valid', activation=tf.nn.relu,
                                                  use_bias=False, name='deconv6')(feats)
        conv6 = tf.keras.layers.Conv2D(filters=33, kernel_size=[3, 3], padding='same', name='conv6')(deconv6)

        return conv6


if __name__ == '__main__':
    a = UpSampleModel(input_shape=(3, 380, 250))