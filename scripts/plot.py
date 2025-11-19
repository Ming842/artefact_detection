"""
Plotting module
"""
from typing import Union

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    """    
    Plotter class for creating and managing plots.
    """
    def __init__(self, mosaic: Union[None, list[list[str]]] = None, fig_title='set Fig Title', figsize = (10, 6)):
        sns.set_theme(context='paper', style='ticks', palette='deep', font_scale=1.2)
        # Make the figure suptitle larger and lower by patching the default suptitle behavior
        _orig_suptitle = plt.Figure.suptitle
        def _suptitle(self, t, *args, **kwargs):
            kwargs.setdefault('y', 0.95)          # move title lower
            kwargs.setdefault('fontsize', 16)     # make title larger
            kwargs.setdefault('fontweight', 'semibold')
            return _orig_suptitle(self, t, *args, **kwargs)
        plt.Figure.suptitle = _suptitle
        self.axes = {}
        if mosaic:
            self.fig, self.axes = plt.subplot_mosaic(mosaic, figsize= figsize)
        else:
            self.fig, self.axes['main'] = plt.subplots(figsize= figsize)
        self.fig_title = fig_title
        self.fig.suptitle(self.fig_title)

    def sync_x_axes(self, axis_keys: list[str]):
        """
        Synchronize x-axes of specified axes.
        """
        if len(axis_keys) < 2:
            return

        first_axis = self.axes[axis_keys[0]]
        for key in axis_keys[1:]:
            self.axes[key].sharex(first_axis)

    def add_plot(self, x, y, axis_key='main', **plot_params):
        """
        Plots data on specified axis with customization options.
        """
        if x is None or y is None:
            print("ERROR: data to plot.")
            return

        self.axes[axis_key].plot(x, y)

        # extract common labels
        xlabel = plot_params.pop('xlabel', '')
        ylabel = plot_params.pop('ylabel', '')
        if xlabel:
            self.axes[axis_key].set_xlabel(xlabel)
        if ylabel:
            self.axes[axis_key].set_ylabel(ylabel)

        # get last plotted line to apply line-level styles
        line = self.axes[axis_key].lines[-1] if self.axes[axis_key].lines else None

        # apply line style parameters if provided
        if line is not None:
            if 'color' in plot_params:
                line.set_color(plot_params.pop('color'))
            if 'linestyle' in plot_params:
                line.set_linestyle(plot_params.pop('linestyle'))
            if 'linewidth' in plot_params:
                line.set_linewidth(plot_params.pop('linewidth'))
            if 'label' in plot_params:
                line.set_label(plot_params.pop('label'))
            if 'alpha' in plot_params:
                line.set_alpha(plot_params.pop('alpha'))
            if 'marker' in plot_params:
                line.set_marker(plot_params.pop('marker'))
            if 'markersize' in plot_params:
                line.set_markersize(plot_params.pop('markersize'))
            if 'markerfacecolor' in plot_params:
                line.set_markerfacecolor(plot_params.pop('markerfacecolor'))
            if 'markeredgecolor' in plot_params:
                line.set_markeredgecolor(plot_params.pop('markeredgecolor'))
            if 'dashes' in plot_params:
                line.set_dashes(plot_params.pop('dashes'))

        # axis-level options
        if 'xlim' in plot_params:
            self.axes[axis_key].set_xlim(plot_params.pop('xlim'))
        if 'ylim' in plot_params:
            self.axes[axis_key].set_ylim(plot_params.pop('ylim'))
        if 'xscale' in plot_params:
            self.axes[axis_key].set_xscale(plot_params.pop('xscale'))
        if 'yscale' in plot_params:
            self.axes[axis_key].set_yscale(plot_params.pop('yscale'))
        if 'grid' in plot_params:
            self.axes[axis_key].grid(plot_params.pop('grid'))
        if plot_params.pop('legend', False):
            self.axes[axis_key].legend()
        if 'legend_loc' in plot_params:
            self.axes[axis_key].legend(loc=plot_params.pop('legend_loc'))
        else:
            self.axes[axis_key].legend(loc='upper right')

    def add_fill_between(self, x, y1, y2=0, axis_key='main', **fill_params):
        """
        Adds filled area between two curves on specified axis.
        """
        self.axes[axis_key].fill_between(x, y1, y1+y2, **fill_params)

    def add_secondary_y_axes(self, original_axis_key, secondary_axis_key, label=''):
        """
        Create secondary y-axis on the main axes.
        """
        twin = self.axes[original_axis_key].twinx()
        self.axes[secondary_axis_key] = twin
        self.axes[secondary_axis_key].grid(False)
        if label:
            self.axes[secondary_axis_key].set_ylabel(label)

    def set_yticks(self, axis_keys: list[str], num_ticks: list[int]):
        """
        Set y-ticks for specified axes.
        """
        for key, ticks in zip(axis_keys, num_ticks):
            self.axes[key].set_yticks(np.linspace(*self.axes[key].get_ybound(), ticks))


    def show(self):
        """
        Shows plots.
        """
        plt.show()
