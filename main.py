"""
Created on Tue Oct 21 11:09:44 2025

@author: 247116 - Ming Dao Caljé 
"""

import numpy as np

import matplotlib.pyplot as plt

import scripts.settings as settings

from scripts.signal_processing import (
    build_signal_and_time)
from scripts.filer import load_database
from scripts.plot import Plotter

from artefact_detection import artefact_detection

def main():
    """
    Main function to perform artefact detection and plot results.
    """

    #initialize settings
    settings.init()

    #load database
    db = load_database(r'decoded_anonymous_data_1.pkl')

    # the data is bunched per time segment, build the full signal and time vector for a set range
    time, signal, time_cco, cco = build_signal_and_time(db, range(0,1000))

    data_properties = artefact_detection(time, signal)

    plot = Plotter([['main']], fig_title='Full Signal with CCO Overlay')
    plot.add_plot(time, signal, axis_key='main',
                color='grey', linewidth = 1, alpha = 0.7,
                xlabel='Time', ylabel='Signal Amplitude (a.u.)',
                label='Raw Signal')

    plot.add_plot(time[data_properties['systolic_peaks']], signal[data_properties['systolic_peaks']],
                  axis_key='main',
                  color='green', linestyle='None', marker='o', label='Systolic Starts')


    # build a color spectrum for the artefact types
    artefact_keys = list(data_properties.get('artefacts', {}).keys())
    n_artefacts = len(artefact_keys)
    if n_artefacts > 0:
        cmap = plt.get_cmap('plasma', n_artefacts)
        colours = [cmap(i) for i in range(n_artefacts)]
    else:
        colours = []

    # plot each artefact type with a color from the spectrum
    for i, key in enumerate(artefact_keys):
        mask = data_properties['artefacts'][key]
        plot.add_plot(np.ma.masked_array(time, ~mask),
                        np.ma.masked_array(signal, ~mask),
                        axis_key='main',
                        color=colours[i] if i < len(colours) else 'red',
                        linewidth=2,
                        alpha = 0.8,
                        label=f'{key.replace("_", " ").title()}',
                        )

    # CCO on secondary y-axis
    plot.add_secondary_y_axes(original_axis_key='main', secondary_axis_key='cco')
    plot.add_plot(time_cco,
                cco,
                axis_key='cco',
                color='orange',
                linewidth = 2,
                alpha = 0.7,
                xlabel='Time',
                ylabel='CCO Signal (l/min)')

    freq_hz = data_properties["fft"]["dominant_frequency"]
    freq_bpm = freq_hz * 60.0
    print(f'{freq_hz:.3f} Hz ({freq_bpm:.1f} bpm) vs {data_properties["heart_rate"]} bpm')

    plot.show()

if __name__ == "__main__":
    main()
