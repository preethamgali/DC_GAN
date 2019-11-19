import tensorflow as tf

fake_img_size = (32,32)
channels = 3

def generator(Z):

  with tf.variable_scope('gen')
    dense_1 = tf.layers.Dense(Z, (1,4*4*512))
    dense_1 = tf.reshape(dense_1, (4,4,512))
    dense_1 = tf.maximum(dens_1, dens_1 * 0.2)
    dens_1 = tf.layers.batch_normalization(dens_1)
    # 4x4x512

    conv_1 = tf.layers.conv2d_transpose(dense_1_act, 256, 5, 2, padding='same')
    conv_1 = tf.maximum(conv_1, conv_1 * 0.2)
    conv_1 = tf.layers.batch_normalization(conv_1)
    # 8x8x256

    conv_2 = tf.layers.conv2d_transpose(conv_1, 128, 5, 2, padding='same')
    conv_2 = tf.maximum(conv_2, conv_2 * 0.2)
    conv_2 = tf.layers.batch_normalization(conv_2)
    # 16x16x128

    conv_3 = tf.layers.conv2d_transpose(conv_2, channels, 5, 2, padding='same')
    fake_img = tf.tanh(conv_3)
    # 32x32x3

    return fake_img

def discriminator(fake_img, reuse):

  with tf.variable_scope('disc',reuse= reusse):
    conv_1 = tf.layers.conv2d(fake_img, 128, 5, 2, padding='same')
    conv_1 =  tf.maximum(conv_1, conv_1 * 0.2)
    conv_1 = tf.layers.batch_normalization(conv_1)
    # 16x16x128

    conv_2 = tf.layers.conv2d(conv_1, 256, 5, 2, padding='same')
    conv_2 = tf.maximum(conv_1, conv_1 * 0.2)
    conv_2 = tf.layers.batch_normalization(conv_1)
    # 8x8x256

    conv_2 = tf.layers.conv2d_transpose(conv_1, 512, 5, 2, padding='same')
    conv_2 = tf.maximum(conv_2, conv_2 * 0.2)
    conv_2 = tf.layers.batch_normalization(conv_2)
    # 4x4x512

    conv_2_reshape = tf.reshape(conv_2, (1,4*4*512))
    # 1x4*4*512

    dense_1 = tf.layers.Dense(conv_2_reshape, (1,256))
    dense_1 = tf.maximum(dense_1)
    dense_1 = tf.layers.batch_normalization(dense_1)

    logits = tf.layers.Dense(dense_1, (1,))

    return logits 
