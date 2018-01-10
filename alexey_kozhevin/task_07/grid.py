#pylint:disable=too-few-public-methods

""" Options and configs. """

from itertools import product

class Pair:
    """ Class for pair value-alias. """
    def __init__(self, value, alias=None):
        """
        Parameters
        ----------
        value : obj
        alias : obj
            if None alias will be equal to value.
        """
        self.value = value
        if alias is None:
            self.alias = value
        else:
            self.alias = alias

    def __repr__(self):
        return str(self.alias) + ': ' + str(self.value)

class Option:
    """ Class for single-parameter option. """
    def __init__(self, parameter, *values):
        """
        Parameters
        ----------
        parameter : Pair
        *values : Pairs
        """
        self.values = values
        self.parameter = parameter

    def alias(self):
        """ Returns alias of the Option. """
        return {self.parameter.alias: [value.alias for value in self.values]}

    def config(self):
        """ Returns config. """
        return {self.parameter.value: [value.value for value in self.values]}

    def __repr__(self):
        return str(self.alias())

    def __mul__(self, other):
        return Grid(self) * Grid(other)

    def __add__(self, other):
        return Grid(self) + Grid(other)

class Config:
    """ Class for config. """
    def __init__(self, config):
        self._config = config

    def alias(self):
        """ Returns alias. """
        return {item[0].alias: item[1].alias for item in self._config}

    def config(self):
        """ Returns values. """
        return {item[0].value: item[1].value for item in self._config}

    def __repr__(self):
        return str(self.alias())

class Grid:
    """ Class for grid of parameters. """
    def __init__(self, grid):
        """
        Parameters
        ----------
        grid: Option, Grid or list of list of Options
        """
        if isinstance(grid, Option):
            self.grid = [[grid]]
        elif isinstance(grid, Grid):
            pass
        else:
            self.grid = grid

    def __len__(self):
        return len(self.grid)

    def __mul__(self, other):
        if isinstance(other, Grid):
            res = list(product(self.grid, other.grid))
            res = [item[0]+item[1] for item in res]
            return Grid(res)
        elif isinstance(other, Option):
            return self * Grid([[other]])

    def __add__(self, other):
        if isinstance(other, Grid):
            return Grid(self.grid + other.grid)
        elif isinstance(other, Option):
            return self + Grid([[other]])

    def alias(self):
        """ Returns alias of the Grid. """
        return [[option.alias() for option in options] for options in self.grid]

    def config(self):
        """ Returns config of the Grid. """
        return [[option.config() for option in options] for options in self.grid]

    def __repr__(self):
        return str(self.alias())

    def __getitem__(self, index):
        return Grid([self.grid[index]])

    def __eq__(self, other):
        return self.config() == other.config()

    def gen_configs(self):
        for item in self.grid:
            keys = [option.parameter for option in item]
            values = [option.values for option in item]
            for parameters in product(*values):
                yield Config(list(zip(keys, parameters)))
