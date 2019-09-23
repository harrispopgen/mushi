#! /usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List

import numpy as np

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class History():
    '''base class piecewise constant history. The first epoch starts at zero,
    and the last epoch extends to infinity

    change_points: epoch change points (times)
    vals: constant values for each epoch (rows)
    '''
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
        if np.any(self.vals <= 0) or np.sum(np.isinf(self.vals)):
            raise ValueError(f'elements of vals must be finite and '
                             'positive')
        self.m = len(self.vals)

    def arrays(self):
        '''return time grid and values in each cell'''
        t = np.concatenate((np.array([0]),
                            self.change_points,
                            np.array([np.inf])))
        return t, self.vals

    def epochs(self):
        '''generator yielding epochs as tuples: (start_time, end_time, value)
        '''
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

    def check_grid(self, other: int):
        '''test if time grid is the same as another instance'''
        if any(self.change_points != other.change_points):
            return False
        else:
            return True

    def plot(self, idxs=None, **kwargs) -> None:
        '''plot the history

        idxs: indices of value column(s) to plot (optional)
        kwargs: key word arguments passed to plt.step
        '''
        t = np.concatenate((np.array([0]), self.change_points))
        if idxs is not None:
            vals = self.vals[:, idxs]
        else:
            vals = self.vals
        plt.step(t, vals, where='post', **kwargs)
        plt.xlabel('$t$')
        if 'label' in kwargs:
            plt.legend()


class η(History):
    '''demographic history

    change_points: epoch change points (times)
    y: vector of constant population sizes in each epoch
    '''
    @property
    def y(self):
        '''read-only alias to vals attribute in base class'''
        return self.vals

    @y.setter
    def y(self, value):
        self.vals = value

    def __post_init__(self):
        super().__post_init__()
        assert len(self.y.shape) == 1, self.y.shape

    def plot(self, **kwargs) -> None:
        super().plot(**kwargs)
        plt.xlabel('$t$')
        plt.ylabel('$η(t)$')
        plt.xscale('symlog')
        plt.yscale('log')
        plt.tight_layout()


@dataclass
class μ(History):
    '''mutation spectrum history

    change_points: epoch change points (times)
    Z: matrix of constant values for each epoch (rows) in each mutation type
       (columns)
    mutation_types: list of mutation type names (default integer names)
    '''
    mutation_types: List[str] = None

    @property
    def Z(self):
        '''read-only alias to vals attribute in base class'''
        return self.vals

    @Z.setter
    def Z(self, value):
        self.vals = value

    def __post_init__(self):
        super().__post_init__()
        assert len(self.Z.shape) == 2, self.Z.shape
        if self.mutation_types is None:
            self.mutation_types = range(1, self.Z.shape[1] + 1)
        assert len(self.mutation_types) == self.Z.shape[1]
        self.mutation_types = pd.Index(self.mutation_types,
                                       name='mutation type')

    def plot(self, idxs=None, **kwargs) -> None:
        super().plot(idxs=idxs, **kwargs)
        plt.xlabel('$t$')
        plt.ylabel('$\\mu(t)$')
        plt.xscale('symlog')
        plt.tight_layout()

    def clustermap(self, **kwargs):
        '''clustermap of k-SFS

        mutation_types: list of column names
        kwargs: additional keyword arguments passed to pd.clustermap
        '''
        t = np.concatenate((np.array([0]), self.change_points))
        Z = self.Z / self.Z.sum(1, keepdims=True)
        Z = Z / Z.mean(0, keepdims=True)
        df = pd.DataFrame(data=Z, index=pd.Index(t, name='$t$'),
                          columns=self.mutation_types)
        g = sns.clustermap(df, row_cluster=False, center=1,
                           metric='correlation',
                           cbar_kws={'label': 'relative mutation intensity'},
                           **kwargs)
        g.ax_heatmap.set_yscale('symlog')
        return g
