"""LinkNet"""

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as Xavier

B_NORM = True
MOMENTUM = 0.1


def encoder_block(inp, training, output_map_size, name):
    """LinkNet encoder block.
    """
    with tf.variable_scope(name): # pylint: disable=not-context-manager
        net = tf.layers.conv2d(inp, output_map_size, (3, 3),
                               strides=(2, 2),
                               padding='SAME',
                               kernel_initializer=Xavier(),
                               name='encoder_conv_1')
        if B_NORM:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm1',
                                                momentum=MOMENTUM)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, output_map_size, (3, 3),
                               padding='SAME',
                               kernel_initializer=Xavier(),
                               name='encoder_conv_2')
        if B_NORM:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm2',
                                                momentum=MOMENTUM)
        net = tf.nn.relu(net)

        shortcut = tf.layers.conv2d(inp, output_map_size, (1, 1),
                                    strides=(2, 2), padding='SAME',
                                    kernel_initializer=Xavier(),
                                    name='encoder_short_1')
        if B_NORM:
            shortcut = tf.layers.batch_normalization(shortcut,
                                                     training=training,
                                                     name='batch-norm-short',
                                                     momentum=MOMENTUM)
        shortcut = tf.nn.relu(shortcut)

        encoder_add = tf.add(net, shortcut, 'encoder_add_1')

        net = tf.layers.conv2d(encoder_add, output_map_size, (3, 3),
                               padding='SAME',
                               kernel_initializer=Xavier(),
                               name='encoder_conv_3')
        if B_NORM:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm3',
                                                momentum=MOMENTUM)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, output_map_size, (3, 3),
                               padding='SAME',
                               kernel_initializer=Xavier(),
                               name='encoder_conv_4')
        if B_NORM:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm4',
                                                momentum=MOMENTUM)
        net = tf.nn.relu(net)

        outp = tf.add(net, encoder_add, 'encoder_add_2')
    return outp


def decoder_block(inp, training, input_map_size, output_map_size, name):
    """LinkNet decoder block.
    """
    with tf.variable_scope(name): # pylint: disable=not-context-manager
        net = tf.layers.conv2d(inp, input_map_size//4, (1, 1),
                               padding='SAME',
                               kernel_initializer=Xavier(),
                               name='decoder_conv_1')
        if B_NORM:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm1',
                                                momentum=MOMENTUM)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d_transpose(net, input_map_size//4, (3, 3),
                                         strides=(2, 2),
                                         padding='SAME',
                                         kernel_initializer=Xavier(),
                                         name='decoder_conv_2'
                                        )
        if B_NORM:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm2',
                                                momentum=MOMENTUM)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, output_map_size, (1, 1),
                               padding='SAME',
                               kernel_initializer=Xavier(),
                               name='decoder_conv_3')
        if B_NORM:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm3',
                                                momentum=MOMENTUM)
        outp = tf.nn.relu(net)

        return outp


def linknet_layers(inp, training, n_classes):
    """LinkNet tf.layers.
    """
    tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)
    with tf.variable_scope('LinkNet'): # pylint: disable=not-context-manager
        net = tf.layers.conv2d(inp, 64, (7, 7),
                               strides=(2, 2),
                               padding='SAME',
                               kernel_initializer=Xavier(),
                               name='conv_1')
        net = tf.layers.max_pooling2d(net, (3, 3), strides=(2, 2), padding='SAME')
        if B_NORM:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm1',
                                                momentum=MOMENTUM)
        net = tf.nn.relu(net)

        enc1 = encoder_block(net, training, 64, '1st_encoder')
        enc2 = encoder_block(enc1, training, 128, '2nd_encoder')
        enc3 = encoder_block(enc2, training, 256, '3rd_encoder')
        enc4 = encoder_block(enc3, training, 512, '4th_encoder')

        dec4 = decoder_block(enc4, training, 512, 256, '4th_decoder')
        net = tf.add(enc3, dec4)

        dec3 = decoder_block(net, training, 256, 128, '3rd_decoder')
        net = tf.add(enc2, dec3)

        dec2 = decoder_block(net, training, 128, 64, '2nd_decoder')
        net = tf.add(enc1, dec2)

        dec1 = decoder_block(net, training, 64, 64, '1st_decoder')

        net = tf.layers.conv2d_transpose(dec1, 32, (3, 3),
                                         strides=(2, 2),
                                         padding='SAME',
                                         kernel_initializer=Xavier(),
                                         name='output_conv_1')

        if B_NORM:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm2',
                                                momentum=MOMENTUM)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, 32, (3, 3),
                               padding='SAME',
                               kernel_initializer=Xavier(),
                               name='output_conv_2')
        if B_NORM:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm3',
                                                momentum=MOMENTUM)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d_transpose(net, n_classes, (2, 2),
                                         strides=(2, 2),
                                         padding='SAME',
                                         kernel_initializer=Xavier(),
                                         name='output_conv_3')
        return net
