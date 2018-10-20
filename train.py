# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler


__author__ = 'njkim@jamonglab.com'


class TimeSeriesData(object):

    def __init__(self, batch_size=128):

        # load data
        # 1st column : time
        # 2nd column : time series values
        # 3rd column : objective (target/category)
        xy = np.genfromtxt('asset/data/serie_target.csv', delimiter=',', dtype=np.float32)
        xy = xy[1:, 1:]

        window = 384 # window size !important : must match with one generator network output dim
        scaler = MinMaxScaler(copy=False)

        # delete zero pad data
        n = ((np.where(np.any(xy, axis=1))[0][-1] + 1) // window) * window

        # normalize data between 0 and 1 for the series column only (0=1st)
        x = scaler.fit_transform(xy[:n,0].reshape(-1, 1))
        y = xy[:n,1] #.reshape(-1, 1)

        # make to matrix
        X = np.asarray([x[i:i+window] for i in range(n-window)])
        Y = np.asarray([y[i:i+window] for i in range(n-window)])
        # shuffle data
        shuf = np.random.choice(n-window, n-window, replace=False)
        X = X[shuf]
        Y = y[shuf]
        X = np.expand_dims(X, axis=2)

        # save to member variable
        self.batch_size = batch_size
        self.X = tf.sg_data._data_to_tensor([X], batch_size, name='train.X')
        self.Y = tf.sg_data._data_to_tensor([Y], batch_size, name='train.Y')
        self.num_batch = X.shape[0] // batch_size


        self.X = tf.to_float(self.X)
        self.Y = tf.to_int32(self.Y)


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size
num_category = 4  # category variable number
num_cont = 1   # continuous variable number
num_dim = 50   # total latent dimension ( category + continuous + noise )

#
# inputs
#

# input tensor ( with QueueRunner )
data = TimeSeriesData(batch_size=batch_size)
x = data.X

# generator labels ( all ones )
y = tf.ones(batch_size, dtype=tf.sg_floatx)

# discriminator labels ( half 1s, half 0s )
y_disc = tf.concat([y, y * 0], 0)


#
# create generator
#

# random class number
z_cat = tf.multinomial(tf.ones((batch_size, num_category), dtype=tf.sg_floatx) / num_category, 1).sg_squeeze().sg_int()

# random seed = random categorical variable + random uniform
z = z_cat.sg_one_hot(depth=num_category).sg_concat(target=tf.random_uniform((batch_size, num_dim-num_category)))

# random continuous variable
z_cont = z[:, num_category:num_category+num_cont]

# category label
label = tf.concat([data.Y, z_cat], 0)

# generator network
with tf.sg_context(name='generator', size=(4,1), stride=(2,1), act='relu', bn=True):
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=48*1*128)
           .sg_reshape(shape=(-1, 48, 1, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=32)
           .sg_upconv(dim=1, act='sigmoid', bn=False))

# add image summary
tf.sg_summary_image(gen)

#
# create discriminator & recognizer
#

# create real + fake image input
xx = tf.concat([x, gen], 0)

with tf.sg_context(name='discriminator', size=(4,1), stride=(2,1), act='leaky_relu'):
    # shared part
    shared = (xx.sg_conv(dim=32)
              .sg_conv(dim=64)
              .sg_conv(dim=128)
              .sg_flatten()
              .sg_dense(dim=1024))
    # shared recognizer part
    recog_shared = shared.sg_dense(dim=128)
    # discriminator end
    disc = shared.sg_dense(dim=1, act='linear').sg_squeeze()
    # categorical recognizer end
    recog_cat = recog_shared.sg_dense(dim=num_category, act='linear')
    # continuous recognizer end
    recog_cont = recog_shared[batch_size:, :].sg_dense(dim=num_cont, act='sigmoid')


#
# loss and train ops
#

loss_disc = tf.reduce_mean(disc.sg_bce(target=y_disc))  # discriminator loss
loss_gen = tf.reduce_mean(disc.sg_reuse(input=gen).sg_bce(target=y))  # generator loss
loss_recog = tf.reduce_mean(recog_cat.sg_ce(target=label)) \
             + tf.reduce_mean(recog_cont.sg_mse(target=z_cont))  # recognizer loss

train_disc = tf.sg_optim(loss_disc + loss_recog, lr=0.0001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen + loss_recog, lr=0.001, category='generator')  # generator train ops


#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_disc, train_disc])[0]  # training discriminator
    l_gen = sess.run([loss_gen, train_gen])[0]  # training generator
    return np.mean(l_disc) + np.mean(l_gen)


# do training
alt_train(log_interval=10, max_ep=50, ep_size=data.num_batch, early_stop=False)

