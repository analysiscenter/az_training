""" File with some useful functions"""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import ewma


plt.style.use('seaborn-poster')
plt.style.use('ggplot')

def draw(first, first_label, second=None, second_label=None, type_data='loss', window=50, bound=None, axis=None):
    """ Draw on graph first and second data.

    The graph shows a comparison of the average values calculated with a 'window'. You can draw one graph
    or create your oun subplots and one of it in 'axis'.

    Parameters
    ----------
    first : list or numpy array
        Have a values to show
    first_label : str
        Name of first data
    second : list or numpy array, optional
        Have a values to show
    second_label : str, optional
        Name of second data
    type_data : str, optional
        Type of data. Example 'loss', 'accuracy'
    window : int, optional
        window width for calculate average value
    bound : list or None
        Bounds to limit graph: [min x, maxis x, min y, maxis y]
    axis : None or element of subplot
        If you want to draw more subplots give the element of subplot """

    firt_ewma = ewma(np.array(first), span=window, adjust=False)
    second_ewma = ewma(np.array(second), span=window, adjust=False) if second else None

    plot = axis or matplotlib.pyplot
    plot.plot(firt_ewma, label='{} {}'.format(first_label, type_data))
    if second_label:
        plot.plot(second_ewma, label='{} {}'.format(second_label, type_data))

    if axis is None:
        plot.xlabel('Iteration', fontsize=16)
        plot.ylabel(type_data, fontsize=16)
    else:
        plot.set_xlabel('Iteration', fontsize=16)
        plot.set_ylabel(type_data, fontsize=16)

    plot.legend(fontsize=14)
    if bound:
        plot.axis(bound)

def separate(layers_names, weights, num_params, bottle):
    """Support fuction that allows yield the data about layer.

    Parameters
    ----------
    layers_names : list of str
        names of layers
    weights : list of str
        weights of layers
    num_params : list or tuple
        number of parameters in each layer
    bottle : bool
        use bottleneck

    Yields
    ------
    names : str
         name of layer
    weights : list
        weights of layer
    num_params : list
        number of parameters in layer
    """
    def _create(layers_names, weights, name, num_params):
        indices = [i for i in range(len(layers_names)) if name in layers_names[i][:8]]
        if name == 'shortcut':
            return np.hstack((weights[indices], [0, 0])), np.hstack((layers_names[indices], [0, 0])), \
                                np.hstack((num_params[indices], [0, 0]))
        return weights[indices], layers_names[indices], num_params[indices]

    if not bottle:
        name = ['layer-0', 'layer-3', 'shortcut']
    else:
        name = ['layer-0', 'layer-3', 'layer-6', 'shortcut']

    data = np.array(_create(layers_names, weights, name[0], num_params))
    for i in name[1:]:
        data = np.vstack((data, _create(layers_names, weights, i, num_params)))
    names, weights, num_params = data[1::3], data[::3], data[2::3]
    for i in range(4):
        yield names[:, i], weights[:, i], num_params[:, i]

def plot_weights(model_names, model_weights, model_params, colors, num_axis, bottleneck=True):
    """Plot distribution of weights

    Parameters
    ----------
    model_names : list or str
        name layers of model
    model_weights : list
        all weights of model
    model_params : list
        number of parameters in layers
    colors : list
        names of colors
    num_axis : list with two elements
        [nrows, ncols] in plt.subplots
    bottleneck : bool
        use bottleneck
        """
    nrows, ncols = num_axis
    _, subplot = plt.subplots(nrows, ncols, sharex='all', figsize=(23, 24))
    subplot = subplot.reshape(-1)
    num_plot = 0
    dict_names = {'bottleneck': {'layer-0': 'first conv 1x1',
                                 'layer-3': 'conv 3x3',
                                 'layer-6': 'second conv 1x1'},
                  'no_bottle': {'layer-0': 'first conv 3x3',
                                'layer-3': 'second conv 3x3'}}
    bottle = 'bottleneck' if bottleneck else 'no_bottle'
    for names, weights, num_params in separate(model_names, model_weights, model_params, bottleneck):
        for name, weight, num in zip(names, weights, num_params):
            if name != 'shortcut' and name != '0':
                name = dict_names[bottle][name]
            subplot[num_plot].set_title('Number of parameners={}\n{}'.format(num, name), fontsize=18)
            if not isinstance(weight, int):
                sns.distplot(weight.reshape(-1), ax=subplot[num_plot], color=colors[int(num_plot % ncols)])
                if num_plot % 1 == 0:
                    dis = (6. / ((weight.shape[2] + weight.shape[3]) * weight.shape[0] * weight.shape[1])) ** 0.5
                    subplot[num_plot].axvline(x=dis, ymax=10, color='k')
                    subplot[num_plot].axvline(x=-dis, ymax=10, color='k')
            subplot[num_plot].set_xlabel('value', fontsize=20)
            subplot[num_plot].set_ylabel('quantity', fontsize=20)
            num_plot += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

def draw_avgpooling(maps, answers, model=True):
    """ Draw maps from GAP

    Parameters
    ----------
    maps : np.array
        all maps from GAP layers
    answers : np.array
        answers to all maps
    model : bool
        se resnet or simple resnet
    """
    col = sns.color_palette("Set2", 8) + sns.color_palette(["#9b59b6", "#3498db"])

    indices = np.array([np.where(answers == i)[0] for i in range(10)])

    filters = np.array([np.mean(maps[indices[i]], axis=0).reshape(-1) for i in range(10)])
    for i in range(10):
        plt.plot(ewma(filters[i], span=350, adjust=False), color=col[i], label=str(i))

    plt.title("Distribution of average pooling in "+("SE ResNet" if model else 'simple ResNet'))
    plt.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Activation', fontsize=18)
    plt.xlabel('Filter index', fontsize=18)
    plt.axis([0, 2060, 0, 1.])
    plt.show()

def axis_draw(freeze_loss, res_loss, src, axis):
    """ Draw graphs to compare models. Theaxis graph shows a comparison of the average
        values calculated with a window in 10 values.
    Args:
        freeze_loss: List with loss value in resnet and freezeout model
        res_loss: List with loss value in clear resnet
        src: List with parameters of model with FreezeOut
        axis: Plt sublot """
    fr_loss = []
    n_loss = []

    for i in range(10, len(res_loss) - 10):
        fr_loss.append(np.mean(freeze_loss[i-10:i+10]))
        n_loss.append(np.mean(res_loss[i-10:i+10]))

    axis.set_title('Freeze model with: LR={} Degree={} It={} Scaled={}'.format(*src))
    axis.plot(fr_loss, label='freeze loss')
    axis.plot(n_loss, label='no freeze loss')
    axis.set_xlabel('Iteration', fontsize=16)
    axis.set_ylabel('Loss', fontsize=16)
    axis.legend(fontsize=14, loc=3)
