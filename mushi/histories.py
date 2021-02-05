r"""History objects for representing demography and MuSH

"""

from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import mushi.composition as cmp


@dataclass
class History():
    """
    Base class piecewise constant history. The first epoch starts at zero,
    and the last epoch extends to infinity

    Args:
        change_points: epoch change points (times)
        vals: constant values for each epoch (rows)
    """
    change_points: np.array
    vals: np.ndarray

    def __post_init__(self):
        if any(np.diff(self.change_points) <= 0) or any(
           np.isinf(self.change_points)) or any(self.change_points <= 0):
            raise ValueError('change_points must be increasing, finite, and '
                             'positive')
        if len(self.vals) != len(self.change_points) + 1:
            raise ValueError(f'len(change_points) = {len(self.change_points)}'
                             f' implies {len(self.change_points) + 1} epochs,'
                             f' but len(vals) = {len(self.vals)}')
        if np.any(self.vals < 0) or np.sum(np.isinf(self.vals)):
            raise ValueError('elements of vals must be finite and nonnegative')
        self.m = len(self.vals)

    def arrays(self):
        """Return time grid and values in each cell"""
        t = np.concatenate((np.array([0]),
                            self.change_points,
                            np.array([np.inf])))
        return t, self.vals

    def epochs(self):
        """Generator yielding epochs as tuples: (start_time, end_time, value)
        """
        for i in range(self.m):
            if i == 0:
                start_time = 0
            else:
                start_time = self.change_points[i - 1]
            if i == self.m - 1:
                end_time = np.inf
            else:
                end_time = self.change_points[i]
            value = self.vals[i]
            yield (start_time, end_time, value)

    def check_grid(self, other: 'History'):
        """Test if time grid is the same as another instance

        Args:
            other: another History object
        """
        if not np.array_equal(self.change_points, other.change_points):
            raise ValueError('histories have different time grids')

    def plot(self, t_gen: float = None, types=None,
             **kwargs) -> List[mpl.lines.Line2D]:
        """Plot the history

        Args:
            t_gen: generation time in years (optional)
            kwargs: key word arguments passed to plt.plot
        """
        t = np.concatenate((np.array([0]), self.change_points))
        if t_gen:
            t *= t_gen
            t_unit = 'years ago'
        else:
            t_unit = 'generations ago'
        if types is not None:
            idxs = [self.mutation_types.get_loc(type) for type in types]
            vals = self.vals[:, idxs]
        else:
            vals = self.vals
        lines = plt.plot(t, vals, **kwargs)
        plt.xlabel(f'$t$ ({t_unit})')
        plt.xscale('log')
        plt.tight_layout()
        return lines


class eta(History):
    """Demographic history

    Args:
        change_points: epoch change points (times)
        y: vector of constant population sizes in each epoch

    Example:

        >>> import mushi
        >>> import numpy as np

        >>> eta = mushi.eta(np.array([1, 10, 100, 1000]),
        ...                 np.array([1000, 100, 100, 1000, 1000]))
    """
    @property
    def y(self):
        """read-only alias to vals attribute in base class"""
        return self.vals

    @y.setter
    def y(self, value):
        self.vals = value

    def __post_init__(self):
        super().__post_init__()
        assert len(self.y.shape) == 1, self.y.shape

    def plot(self, **kwargs) -> None:
        super().plot(**kwargs)
        plt.ylabel('$\\eta(t)$')
        plt.yscale('log')
        plt.tight_layout()


@dataclass
class mu(History):
    """Mutation spectrum history (MuSH)

    Args:
        change_points: epoch change points (times)
        Z: matrix of constant values for each epoch (rows) in each mutation type
           (columns)
        mutation_types: list of mutation type names (optional ``None`` element)

    Example:

        >>> import mushi
        >>> import numpy as np

        >>> mu = mushi.mu(np.array([1, 10, 100, 1000]),
        ...               np.ones((5, 2)),
        ...               ['TCC>TTC', 'GAA>GGA'])
    """
    mutation_types: List[str]

    @property
    def Z(self):
        """read-only alias to vals attribute in base class"""
        return self.vals

    @Z.setter
    def Z(self, value):
        self.vals = value

    def __post_init__(self):
        super().__post_init__()
        # if mutation rate vector instead of matrix, promote to matrix
        if len(self.Z.shape) == 1:
            self.Z = self.Z[:, np.newaxis]
        assert len(self.Z.shape) == 2, self.Z.shape
        assert len(self.mutation_types) == self.Z.shape[1]
        self.mutation_types = pd.Index(self.mutation_types,
                                       name='mutation type')

    def plot(self, types: List[str] = None, clr: bool = False,
             **kwargs) -> None:
        """Plot history

        Args:
            types: list of mutation types to plot (default all)
            clr: flag to normalize to total mutation intensity and display as
                 centered log ratio transform
        """
        lines = super().plot(types=types, **kwargs)
        if clr:
            Z = cmp.clr(self.Z)
            if types is None:
                idxs = range(len(lines))
            else:
                idxs = [self.mutation_types.get_loc(type) for type in types]
            for idx, line in zip(idxs, lines):
                line.set_ydata(Z[:, idx])
            # recompute the ax.dataLim
            ax = plt.gca()
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()
            # plt.ylabel('relative mutation intensity')
            plt.ylabel('mutation intensity composition\n(CLR transformed)')
        else:
            plt.ylabel('$\\mu(t)$')
        plt.tight_layout()

    def plot_cumulative(self, t_gen: float = None, clr: bool = False,
                        **kwargs) -> None:
        r"""Plot the cumulative mutation rate, like a Muller plot

        Args:
            t_gen: generation time in years (optional)
            clr: flag to normalize to total mutation intensity and display as
                 centered log ratio transform
            kwargs: key word arguments passed to ``plt.fill_between``
        """
        t = np.concatenate((np.array([0]), self.change_points))
        if t_gen:
            t *= t_gen
            t_unit = 'years ago'
        else:
            t_unit = 'generations ago'
        Z = np.cumsum(self.Z, axis=1)
        if clr:
            Z = cmp.clr(Z)
        for j in range(Z.shape[1]):
            plt.fill_between(t, Z[:, j - 1] if j else 0, Z[:, j], **kwargs)
        plt.xlabel(f'$t$ ({t_unit})')
        plt.ylabel('$\\mu(t)$')
        plt.xscale('log')
        plt.tight_layout()

    def clustermap(self, t_gen: float = None, **kwargs) -> None:
        r"""Clustermap of compositionally centralized MuSH

        Args:
            t_gen: generation time in years (optional)
            kwargs: additional keyword arguments passed to ``pd.clustermap``
        """
        t = np.concatenate((np.array([0]), self.change_points))
        if t_gen:
            t *= t_gen
            t_unit = 'years ago'
        else:
            t_unit = 'generations ago'
        Z = cmp.centralize(self.Z)
        label = 'mutation intensity\nperturbation'
        df = pd.DataFrame(data=Z, index=pd.Index(t, name=f'$t$ ({t_unit})'),
                          columns=self.mutation_types)
        g = sns.clustermap(df, row_cluster=False,
                           cbar_kws={'label': label},
                           **kwargs)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(),
                                     fontsize=9, family='monospace')
