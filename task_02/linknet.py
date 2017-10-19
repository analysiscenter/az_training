"LinkNet as TFModel"
import tensorflow as tf
from dataset.dataset.models.tf import TFModel

class LinkNetModel(TFModel):
    "LinkNet as TFModel"
    def _build(self, *args, **kwargs):
        "build for LinkNet"
        images_shape = list(self.get_from_config('images_shape'))
        n_classes = self.get_from_config('n_classes')
        b_norm = self.get_from_config('b_norm', True)
        momentum = self.get_from_config('momentum', 0.9)

        input_ph = tf.placeholder(tf.float32, shape=[None] + images_shape, name='input_image')

        if len(images_shape) == 2:
            input_ph = tf.reshape(input_ph, [-1] + images_shape + [1])
            mask_ph = tf.placeholder(tf.float32, shape=[None] + images_shape + [n_classes], name='targets')
        elif len(images_shape) == 3:
            mask_ph = tf.placeholder(tf.float32, shape=[None] + images_shape[:-1] + [n_classes], name='targets')           
        else:
            raise ValueError('len(images_shape) must be 2 or 3')

        training_ph = tf.placeholder(tf.bool, shape=[], name='bn_mode')
        model_output = linknet_layers(input_ph, training_ph, n_classes, b_norm, momentum)
        predictions = tf.identity(model_output, name='predictions')
        y_pred_softmax = tf.nn.softmax(predictions, name='predicted_prob')
        tf.cast(tf.argmax(y_pred_softmax, axis=3), tf.float32, name='mask_prediction')



def encoder_block(inp, training, output_map_size, name, b_norm, momentum):
    """LinkNet encoder block.
    """
    with tf.variable_scope(name): # pylint: disable=not-context-manager
        net = tf.layers.conv2d(inp, output_map_size, (3, 3),
                               strides=(2, 2),
                               padding='SAME',
                               name='encoder_conv_1')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm1',
                                                momentum=momentum)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, output_map_size, (3, 3),
                               padding='SAME',
                               name='encoder_conv_2')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm2',
                                                momentum=momentum)
        net = tf.nn.relu(net)

        shortcut = tf.layers.conv2d(inp, output_map_size, (1, 1),
                                    strides=(2, 2), padding='SAME',
                                    name='encoder_short_1')
        if b_norm:
            shortcut = tf.layers.batch_normalization(shortcut,
                                                     training=training,
                                                     name='batch-norm-short',
                                                     momentum=momentum)
        shortcut = tf.nn.relu(shortcut)

        encoder_add = tf.add(net, shortcut, 'encoder_add_1')

        net = tf.layers.conv2d(encoder_add, output_map_size, (3, 3),
                               padding='SAME',
                               name='encoder_conv_3')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm3',
                                                momentum=momentum)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, output_map_size, (3, 3),
                               padding='SAME',
                               name='encoder_conv_4')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm4',
                                                momentum=momentum)
        net = tf.nn.relu(net)

        outp = tf.add(net, encoder_add, 'encoder_add_2')
    return outp


def decoder_block(inp, training, input_map_size, output_map_size, name, b_norm, momentum):
    """LinkNet decoder block.
    """
    with tf.variable_scope(name): # pylint: disable=not-context-manager
        net = tf.layers.conv2d(inp, input_map_size//4, (1, 1),
                               padding='SAME',
                               name='decoder_conv_1')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm1',
                                                momentum=momentum)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d_transpose(net, input_map_size//4, (3, 3),
                                         strides=(2, 2),
                                         padding='SAME',
                                         name='decoder_conv_2'
                                        )
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm2',
                                                momentum=momentum)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, output_map_size, (1, 1),
                               padding='SAME',
                               name='decoder_conv_3')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm3',
                                                momentum=momentum)
        outp = tf.nn.relu(net)

        return outp


def linknet_layers(inp, training, n_classes, b_norm, momentum):
    """LinkNet tf.layers.
    """
    with tf.variable_scope('LinkNet'): # pylint: disable=not-context-manager
        net = tf.layers.conv2d(inp, 64, (7, 7),
                               strides=(2, 2),
                               padding='SAME',
                               name='conv_1')
        net = tf.layers.max_pooling2d(net, (3, 3), strides=(2, 2), padding='SAME')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm1',
                                                momentum=momentum)
        net = tf.nn.relu(net)

        enc1 = encoder_block(net, training, 64, '1st_encoder', b_norm, momentum)
        enc2 = encoder_block(enc1, training, 128, '2nd_encoder', b_norm, momentum)
        enc3 = encoder_block(enc2, training, 256, '3rd_encoder', b_norm, momentum)
        enc4 = encoder_block(enc3, training, 512, '4th_encoder', b_norm, momentum)

        dec4 = decoder_block(enc4, training, 512, 256, '4th_decoder', b_norm, momentum)
        net = tf.add(enc3, dec4)

        dec3 = decoder_block(net, training, 256, 128, '3rd_decoder', b_norm, momentum)
        net = tf.add(enc2, dec3)

        dec2 = decoder_block(net, training, 128, 64, '2nd_decoder', b_norm, momentum)
        net = tf.add(enc1, dec2)

        dec1 = decoder_block(net, training, 64, 64, '1st_decoder', b_norm, momentum)

        net = tf.layers.conv2d_transpose(dec1, 32, (3, 3),
                                         strides=(2, 2),
                                         padding='SAME',
                                         name='output_conv_1')

        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm2',
                                                momentum=momentum)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, 32, (3, 3),
                               padding='SAME',
                               name='output_conv_2')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm3',
                                                momentum=momentum)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d_transpose(net, n_classes, (2, 2),
                                         strides=(2, 2),
                                         padding='SAME',
                                         name='output_conv_3')
        return net
