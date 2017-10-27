""" LinkNet as TFModel """
import tensorflow as tf
from dataset.dataset.models.tf.layers import conv_block
from basemodels import NetworkModel

class UNetModel(NetworkModel):
    """LinkNet as TFModel"""
    def _build(self, inputs, *args, **kwargs):

        n_classes = self.num_channels('masks')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)

        conv = {'data_format': data_format}
        batch_norm = {'momentum': 0.1}

        kwargs = {'conv': conv, 'batch_norm': batch_norm}

        layout = 'cnacnap' if b_norm else 'cacap'

        net = inputs['images']
        encoder_outputs = []
        unet_filters = [64, 128, 256, 512]
        axis = dim+1 if data_format == 'channels_last' else 1

        for i, filters in enumerate(unet_filters):
            net = conv_block(dim, net, filters, 3, layout, 'encoder-'+str(i),
                             pool_size=2, **kwargs)
            encoder_outputs.append(net)

        layout = 'cnacna' if b_norm else 'caca'

        net = conv_block(dim, net, 1024, 3, layout, 'middle-block', **kwargs)
        net = conv_block(dim, net, 512, 2, 't', 'middle-block-out', 2, **kwargs)


        for i, filters in enumerate(unet_filters[:0:-1]):
            net = tf.concat([encoder_outputs[-i-2], net], axis=axis)
            net = conv_block(dim, net, filters, 3, layout, 'decoder-block-'+str(i), **kwargs)
            net = conv_block(dim, net, filters // 2, 2, 't', 'decoder-block-out-'+str(i), 2, **kwargs)

        net = conv_block(dim, net, 64, 3, layout, 'decoder-block-3', **kwargs)
        net = conv_block(dim, net, n_classes, 1, layout, 'decoder-block-3-out', **kwargs)

        tf.identity(net, 'predictions')
