import sys

import numpy as np
import matplotlib.pyplot as plt


sys.path.append("..//..")
from dataset.models.tf.resnet import ResNet
from dataset import DatasetIndex, Dataset, F, V, B, Pipeline


def plot_results(ordered_keys, stats, time_dict, window=10, zoom=20, name='Loss'):
    plt.style.use('seaborn-poster')
    cols = ['m', 'gold', 'r', 'g', 'lawngreen', 'b' , 'c', 'midnightblue']

    fig, axes = plt.subplots(nrows=2, ncols=1)
    for index, key in enumerate(ordered_keys):
        current_time = np.cumsum(time_dict[key][2:])
        current_loss = stats[key]
        smoothed_current = []
        for i in range(12, len(current_loss) - window):
            smoothed_current.append(np.mean(current_loss[i-window:i+window]))
        axes[0].plot(current_time[:len(smoothed_current) - 2], smoothed_current[2:], \
                     c=cols[index], label = 'k=' + str(key[-1]), alpha=0.8)
        axes[1].plot(current_time[:zoom], smoothed_current[:zoom], c=cols[index], \
                     label= 'k=' + str(key[-1]), alpha=0.8)

    fig.suptitle(name + 'plot for different ResNet architectures')
    axes[0].legend()
    axes[0].set_xlabel('time')
    axes[0].set_ylabel('loss')

    axes[1].legend()
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('loss')
    plt.show()


    def all_parameters_count(ppl_dict):
        def parameters_count(ppl):
            all_layers = ppl.get_model_by_name('MyResNet').graph.get_collection('trainable_variables')
            n_parameters = 0
            for layer in all_layers:
                current_n = 1
                for dim_shape in (layer.get_shape().as_list()):
                    current_n *= dim_shape
                n_parameters += current_n
            return (n_parameters)

        for key, ppl in ppl_dict.items():
            n_parameters = parameters_count(ppl)
            print('key= ', key, 'n_parameters= ', n_parameters)


    def search(template_ppl, default_config, widening_factors, num_blocks, filters, test_losses, pipelines_dict, time_dict, name='0'):
        for index, wid_factor in enumerate(widening_factors):      
            for i in range(15):
                if (wid_factor == 4) and i > 3:
                    break
                num_blocks[index] = [i, i, i]

                config = {**default_config,
                          'body/num_blocks': num_blocks[index],
                          'body/block/width_factor': wid_factor,
                          'body/filters': filters}

                ppl = ((template_ppl << mnistset.train)
                                 .add_noise()
                                 .init_variable('start_time', init_on_each_run=0)
                                 .init_variable('time_history', init_on_each_run=list)
                                 .init_variable('loss_history', init_on_each_run=list)
                                 .init_model('dynamic', ResNet, 'MyResNet', config)
                                 .update_start_time()
                                 .train_model('MyResNet', fetches=['loss', 'accuracy'], feed_dict={'images': B('images'),
                                                                                                   'labels': B('labels')},
                                             save_to=[V('loss_history'), V('acc_history')], mode='a')
                                 .update_time_history())

                ppl.next_batch(100, n_epochs=None, shuffle=True)
                all_parameters_count({(str(i) + '+' + str(wid_factor)) : ppl})
            print('next')
        return all_parameters_count(pipelines_dict)
