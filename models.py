from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim


conv = functools.partial(slim.conv2d, activation_fn=None)
deconv = functools.partial(slim.conv2d_transpose, activation_fn=None)
relu = tf.nn.relu
lrelu = functools.partial(ops.leak_relu, leak=0.2)


def discriminator(img, scope, df_dim=64, reuse=False, train=True):

    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)

    with tf.variable_scope(scope + '_discriminator', reuse=reuse):
        # h0 is (128 x 128 x df_dim)
        h0 = lrelu(conv(img, df_dim, 4, 2, scope='h0_conv'))
        h1 = lrelu(bn(conv(h0, df_dim * 2, 4, 2, scope='h1_conv'),
                      scope='h1_bn'))  # h1 is (64 x 64 x df_dim*2)
        h2 = lrelu(bn(conv(h1, df_dim * 4, 4, 2, scope='h2_conv'),
                      scope='h2_bn'))  # h2 is (32x 32 x df_dim*4)
        h3 = lrelu(bn(conv(h2, df_dim * 8, 4, 1, scope='h3_conv'),
                      scope='h3_bn'))  # h3 is (32 x 32 x df_dim*8)
        h4 = conv(h3, 1, 4, 1, scope='h4_conv')  # h4 is (32 x 32 x 1)

        return h4


def generator_resnet(img, scope, gf_dim=64, reuse=False, train=True):

    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)

    def residule_block(x, dim, scope='res'):
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = relu(bn(conv(y, dim, 3, 1, padding='VALID',
                         scope=scope + '_conv1'), scope=scope + '_bn1'))
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = bn(conv(y, dim, 3, 1, padding='VALID',
                    scope=scope + '_conv2'), scope=scope + '_bn2')
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=reuse):
        c0 = tf.pad(img, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = relu(bn(conv(c0, gf_dim, 7, 1, padding='VALID',
                          scope='c1_conv'), scope='c1_bn'))
        c2 = relu(
            bn(conv(c1, gf_dim * 2, 3, 2, scope='c2_conv'), scope='c2_bn')
        )
        c3 = relu(
            bn(conv(c2, gf_dim * 4, 3, 2, scope='c3_conv'), scope='c3_bn')
        )

        r1 = residule_block(c3, gf_dim * 4, scope='r1')
        r2 = residule_block(r1, gf_dim * 4, scope='r2')
        r3 = residule_block(r2, gf_dim * 4, scope='r3')
        r4 = residule_block(r3, gf_dim * 4, scope='r4')
        r5 = residule_block(r4, gf_dim * 4, scope='r5')
        r6 = residule_block(r5, gf_dim * 4, scope='r6')
        r7 = residule_block(r6, gf_dim * 4, scope='r7')
        r8 = residule_block(r7, gf_dim * 4, scope='r8')
        r9 = residule_block(r8, gf_dim * 4, scope='r9')

        d1 = relu(
            bn(deconv(r9, gf_dim * 2, 3, 2, scope='d1_dconv'), scope='d1_bn')
        )
        d2 = relu(
            bn(deconv(d1, gf_dim, 3, 2, scope='d2_dconv'), scope='d2_bn')
        )
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv(d2, 3, 7, 1, padding='VALID', scope='pred_conv')
        pred = tf.nn.tanh(pred)

        return pred


# def generator_unet(image, options, reuse=False, name="generator"):
def generator_unet(img, scope, gf_dim=64, reuse=False, train=True):

    if train:
        dropout_rate = 0.5
    else:
        dropout_rate = 1.0

    with tf.variable_scope(scope + '_generator', reuse=reuse):
        # image is (256 x 256 x input_c_dim)
        e1 = ops.instance_norm(ops.conv2d(img, gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = ops.instance_norm(ops.conv2d(
            lrelu(e1), gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = ops.instance_norm(ops.conv2d(
            lrelu(e2), gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = ops.instance_norm(ops.conv2d(
            lrelu(e3), gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = ops.instance_norm(ops.conv2d(
            lrelu(e4), gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = ops.instance_norm(ops.conv2d(
            lrelu(e5), gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = ops.instance_norm(ops.conv2d(
            lrelu(e6), gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = ops.instance_norm(ops.conv2d(
            lrelu(e7), gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv(tf.nn.relu(e8), gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([ops.instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv(tf.nn.relu(d1), gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([ops.instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv(tf.nn.relu(d2), gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([ops.instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv(tf.nn.relu(d3), gf_dim*8, name='g_d4')
        d4 = tf.concat([ops.instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv(tf.nn.relu(d4), gf_dim*4, name='g_d5')
        d5 = tf.concat([ops.instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv(tf.nn.relu(d5), gf_dim*2, name='g_d6')
        d6 = tf.concat([ops.instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv(tf.nn.relu(d6), gf_dim, name='g_d7')
        d7 = tf.concat([ops.instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        output_c_dim = 3
        d8 = deconv(tf.nn.relu(d7), output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)
