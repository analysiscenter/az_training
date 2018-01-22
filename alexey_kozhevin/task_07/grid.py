#pylint:disable=too-few-public-methods

""" Options and configs. """

from dataset import Config

from itertools import product
import collections

class KV:
    """ Class for value and alias. """
    def __init__(self, value, alias=None):
        """
        Parameters
        ----------
        value : obj
        alias : obj
            if None alias will be equal to value.
        """
        if isinstance(value, KV):
            self.value = value.value
            self.alias = value.alias
        else:
            self.value = value
            if alias is None:
                self.alias = value
            else:
                self.alias = alias

    def __repr__(self):
        return 'KV(' + str(self.alias) + ': ' + str(self.value) + ')'

class Option:
    """ Class for single-parameter option. """
    def __init__(self, parameter, values):
        """
        Parameters
        ----------
        parameter : KV or obj
        values : list of KV or obj
        """
        self.parameter = KV(parameter)
        self.values = [KV(value) for value in values]

    def alias(self):
        """ Returns alias of the Option. """
        return {self.parameter.alias: [value.alias for value in self.values]}

    def option(self):
        """ Returns config. """
        return {self.parameter.value: [value.value for value in self.values]}

    def __repr__(self):
        return 'Option(' + str(self.alias()) + ')'

    def __mul__(self, other):
        return Grid(self) * Grid(other)

    def __add__(self, other):
        return Grid(self) + Grid(other)

    def gen_configs(self):
        """ Returns Configs created from the option. """
        grid = Grid(self)
        return grid.gen_configs()

class ConfigAlias:
    """ Class for config. """
    def __init__(self, config):
        """
        Parameters
        ----------
        config : list of (key, value)
            keys and values are KV
        """
        self._config = config

    def alias(self, as_string=False, delim='_'):
        """ Returns alias. """
        config_alias = {item[0].alias: item[1].alias for item in self._config}
        if as_string is False:
            return config_alias
        else:
            config_alias = collections.OrderedDict(sorted(config_alias.items()))
            return delim.join([str(key)+'-'+str(value) for key, value in config_alias.items()])

    def config(self):
        """ Returns values. """
        return Config({item[0].value: item[1].value for item in self._config})

    def __repr__(self):
        return 'ConfigAlias(' + str(self.alias()) + ')'

class Grid:
    """ Class for grid of parameters. """
    def __init__(self, grid):
        """
        Parameters
        ----------
        grid: Option, Grid or list of lists of Options
        """
        if isinstance(grid, Option):
            self._grid = [[grid]]
        elif isinstance(grid, Grid):
            self._grid = grid._grid
        else:
            self._grid = grid

    def alias(self):
        """ Returns alias of Grid. """
        return [[option.alias() for option in options] for options in self._grid]

    def grid(self):
        """ Returns config of Grid. """
        return [[option.option() for option in options] for options in self._grid]

    def __len__(self):
        return len(self._grid)

    def __mul__(self, other):
        if isinstance(other, Grid):
            res = list(product(self._grid, other._grid))
            res = [item[0] + item[1] for item in res]
            return Grid(res)
        elif isinstance(other, Option):
            return self * Grid([[other]])

    def __add__(self, other):
        if isinstance(other, Grid):
            return Grid(self._grid + other._grid)
        elif isinstance(other, Option):
            return self + Grid([[other]])

    def __repr__(self):
        return 'Grid(' + str(self.alias()) + ')'

    def __getitem__(self, index):
        return Grid([self._grid[index]])

    def __eq__(self, other):
        return self.grid() == other.grid()

    def gen_configs(self):
        """ Generate Configs from grid. """
        for item in self._grid:
            keys = [option.parameter for option in item]
            values = [option.values for option in item]
            for parameters in product(*values):
                yield ConfigAlias(list(zip(keys, parameters)))
