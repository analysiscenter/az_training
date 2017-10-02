import tensorflow as tf

def conv_mpool_bnorm_activation(scope, input_layer, n_channels=2, mpool=False, bnorm=True, training=None, kernel_conv=(5, 5),
                                stride_conv=(1, 1), kernel_pool=(2, 2), stride_pool=(2, 2),  activation=tf.nn.relu):
    """ Conv -> mpooling (optional) -> activation layer
    """
    with tf.variable_scope(scope):
        # infer input_nchannels
        inp_channels = input_layer.shape.as_list()[-1]

        # define var for conv-filter
        filter_shape = tuple(kernel_conv) + (inp_channels, ) + (n_channels, )
        filter_weights = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.01), name='weights')

        # bias
        bias = tf.Variable(tf.zeros(shape=[n_channels]), name='bias')

        # apply the filter
        strides = (1, ) + tuple(stride_conv) + (1, )
        output = tf.nn.conv2d(input=input_layer, filter=filter_weights, strides=strides, padding='SAME')

        # bias
        output = output + bias

        # apply mpooling if needed
        if mpool:
            ksize = (1, ) + tuple(kernel_pool) + (1, )
            strides = (1, ) + tuple(stride_pool) + (1, )
            output = tf.nn.max_pool(output, ksize=ksize, strides=strides, padding='SAME')

        # bnorm if needed
        if bnorm:
            output = tf.layers.batch_normalization(output, training=training, name='batch-norm')

        return tf.identity(activation(output), name='output')

def fc_layer(scope, input_layer, n_outs):
    """ Build fully-connected layer with n_outs outputs
    
    Args:
        input_layer: input layer
        n_outs: dim of output tensor
    Return:
        output layer (tf-tensor)
    """
    with tf.variable_scope(scope):
        n_ins = input_layer.shape.as_list()[-1]
        W = tf.Variable(tf.random_normal([n_ins, n_outs]), name='weights')
        b = tf.Variable(tf.zeros([n_outs]), name='bias')
        output = tf.nn.xw_plus_b(input_layer, W, b, name='output')
        return output
