import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2

# # download dataset from kaggle
# !mkdir /root/.kaggle
# !echo '{"username":"preethamgali","key":"*************************"}' > kaggle.json
# !!cp kaggle.json ~/.kaggle/
# !chmod 600 /root/.kaggle/kaggle.json
# !kaggle datasets download -d kvpratama/pokemon-images-dataset
# # unzipping data
# filename = "/content/pokemon-images-dataset.zip"
# from zipfile import ZipFile
# archive = ZipFile(filename, 'r')
# archive.extractall()


z_latent_size = 1000
image_shape = (256,256,4)

def get_data():
  directory = '/content/pokemon/'
  data = []
  img_files = os.listdir(directory)
  for img in img_files:
    if img.endswith('.png'):
      d = mpimg.imread(directory+img)
      d = d*2 -1
      d = cv2.resize(d,(image_shape[0],image_shape[1]))
      data.append(d)
  return np.asarray(data)

def gan_inputs():
  # single astrisk ie., --> * is used to unpack  
    img = tf.placeholder(tf.float32, (None,*image_shape))
    z = tf.placeholder(tf.float32, (None, z_latent_size))
    return z, img

def generator(Z,reuse=False, is_train= True):

  with tf.variable_scope('gen', reuse=reuse):
    # z = 100

    dense_1 = tf.layers.dense(inputs= Z, units= 4*4*512)
    dense_1 = tf.maximum(dense_1, dense_1 * 0.2)
    dense_1 = tf.reshape(tensor=dense_1, shape=(-1,4,4,512))
    # 4x4x512

    conv_1 = tf.layers.conv2d_transpose(inputs= dense_1, filters= 256, kernel_size= 5, strides= 2, padding='same')
    conv_1 = tf.layers.batch_normalization(conv_1, training=is_train)
    conv_1 = tf.maximum(conv_1, conv_1 * 0.2)
    # 8x8x256

    conv_2 = tf.layers.conv2d_transpose(inputs= conv_1, filters= 128, kernel_size= 5, strides= 2, padding='same')
    conv_2 = tf.layers.batch_normalization(conv_2, training=is_train)
    conv_2 = tf.maximum(conv_2, conv_2 * 0.2)
    # 16x16x128

    conv_3 = tf.layers.conv2d_transpose(inputs= conv_2, filters= 128, kernel_size= 5, strides= 2, padding='same')
    conv_3 = tf.layers.batch_normalization(conv_3, training=is_train)
    conv_3 = tf.maximum(conv_3, conv_3 * 0.2)
    # 32x32x128

    conv_4 = tf.layers.conv2d_transpose(inputs= conv_3, filters= 128, kernel_size= 5, strides= 2, padding='same')
    conv_4 = tf.layers.batch_normalization(conv_4, training=is_train)
    conv_4 = tf.maximum(conv_4, conv_4 * 0.2)
    # 64x64x128

    conv_5 = tf.layers.conv2d_transpose(inputs= conv_4, filters= 64, kernel_size= 5, strides= 2, padding='same')
    conv_5 = tf.layers.batch_normalization(conv_5, training=is_train)
    conv_5 = tf.maximum(conv_5, conv_5 * 0.2)
    # 128x128x64

    conv_6 = tf.layers.conv2d_transpose(inputs= conv_5, filters= 4, kernel_size= 5, strides= 2, padding='same')
    fake_img = tf.tanh(conv_6)
    # 256x256x4

  return fake_img

def discriminator(img, reuse=False, is_train=True):
  
  with tf.variable_scope('disc',reuse= reuse):
      # img = 256x256x4
      conv_1 = tf.layers.conv2d(inputs= img, filters= 64, kernel_size= 5, strides= 2, padding='same')
      conv_1 =  tf.maximum(conv_1, conv_1 * 0.2)
      # 128x128x64

      conv_2 = tf.layers.conv2d(inputs= conv_1, filters= 64, kernel_size= 5, strides= 2, padding='same')
      conv_2 = tf.layers.batch_normalization(conv_2, training=is_train)
      conv_2 = tf.maximum(conv_2, conv_2 * 0.2)
      # 64x64x64

      conv_3 = tf.layers.conv2d(inputs= conv_2, filters= 128, kernel_size= 5, strides= 2, padding='same')
      conv_3 = tf.layers.batch_normalization(conv_3, training=is_train)
      conv_3 = tf.maximum(conv_3, conv_3 * 0.2)
      # 32x32x128

      conv_4 = tf.layers.conv2d(inputs= conv_3, filters= 128, kernel_size= 5, strides= 2, padding='same')
      conv_4 = tf.layers.batch_normalization(conv_4, training=is_train)
      conv_4 = tf.maximum(conv_4, conv_4 * 0.2)
      # 16x16x128

      conv_5 = tf.layers.conv2d_transpose(inputs= conv_4, filters= 256, kernel_size= 5, strides= 2, padding='same')
      conv_5 = tf.layers.batch_normalization(conv_5, training=is_train)
      conv_5 = tf.maximum(conv_5, conv_5 * 0.2)
      # 8x8x256

      conv_6 = tf.layers.conv2d_transpose(inputs= conv_5, filters= 512, kernel_size= 5, strides= 2, padding='same')
      conv_6 = tf.layers.batch_normalization(conv_6, training=is_train)
      conv_6 = tf.maximum(conv_6, conv_6 * 0.2)
      # 4x4x512

      conv_6_reshape = tf.reshape(conv_6, shape= (-1,4*4*512))
      # 4*4*512

      dense_1 = tf.layers.dense(conv_6_reshape, units= 256)
      dense_1 = tf.maximum(dense_1, dense_1 * 0.2)

      logits = tf.layers.dense(dense_1, 1)
      tf.print(logits)

  return logits

Z, real_img = gan_inputs()

generated_img = generator(Z,reuse=False, is_train=True)
generate_img = generator(Z,reuse=True, is_train=False)
logits_real_img = discriminator(real_img,reuse=False, is_train=True)
logits_fake_img = discriminator(generated_img, reuse= True, is_train=True)

loss_real_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real_img), logits= logits_real_img))
loss_fake_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(logits_fake_img), logits= logits_fake_img))
d_loss = loss_real_img + loss_fake_img

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_fake_img), logits= logits_fake_img))

t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if 'gen' in var.name]
d_vars = [var for var in t_vars if 'disc' in var.name]

train_g = tf.train.AdamOptimizer().minimize(g_loss, var_list = g_vars)
train_d = tf.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)

with tf.device('/GPU:0'):
  data_set = get_data()
  batch_size = 100
  epochs = 5000
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  batches = [data_set[i:i+batch_size] for i in range(0,data_set.shape[0],batch_size)]
  for e in range(epochs):
    for batch in batches:
      z_latent = np.random.uniform(-1,1,size=(batch_size, z_latent_size))
      sess.run(train_d , feed_dict= {Z:z_latent, real_img:batch})
      sess.run(train_g, feed_dict= {Z:z_latent})

    if e%(epochs/10) == 0:
      d_error = sess.run(d_loss,feed_dict= {Z:z_latent, real_img:batch})
      g_error = sess.run(g_loss,feed_dict= {Z:z_latent})
      print('e:',e,
          'd_error:',np.mean(d_error),
          'g_error:',np.mean(g_error))

z_latent = np.random.uniform(-1,1,size=(1, z_latent_size))
genrated_img = sess.run(generate_img, feed_dict= {Z:z_latent}).reshape(image_shape)
plt.imshow(genrated_img)
