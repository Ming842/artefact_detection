"""
Artefact detection module.
"""

import numpy as np

import scripts.settings as settings

from scripts.signal_processing import (
    calculate_fft,
    build_mask,
    calc_heart_rate,
    find_dips,
    mask_slinger_artefact,
    mask_calibration_artefacts,
    mask_flush_artefacts,
    mask_block_artefacts,
    solve_artefact_priority
)

def artefact_detection(time, signal) -> dict:
    """
    Perform artefact detection on the given signal and store results in data_properties.
    """

    # initialize data properties dictionary
    data_properties = {}

    # calculate the heart rate from the signal and systolic_peaks
    heart_rate, peak_sys_indices = calc_heart_rate(time, signal)

    # store heart rate in data properties
    data_properties['heart_rate'] = heart_rate

    # store systolic peaks in data properties
    sys_peaks_mask = build_mask(peak_sys_indices, len(signal)) #buils boolean mask
    data_properties['systolic_peaks'] = sys_peaks_mask

    # calculate mean signal for plotting
    mean_signal = np.convolve(signal, np.ones(20)/20, mode='same')
    data_properties['mean_signal'] = mean_signal

    # perform FFT analysis
    frequency, amplitude = calculate_fft(signal)

    # store FFT results in a clearer structure and add the dominant frequency for convenience-
    data_properties['fft'] = {
        'frequency': frequency,
        'amplitude': amplitude,
        'dominant_frequency': frequency[np.argmax(amplitude)] if amplitude.size > 0 else None
    }

    # store diastolic dips in data properties
    start_sys_indices = find_dips(signal)
    dips_mask = build_mask(start_sys_indices, len(signal)) #buils boolean mask
    data_properties['systolic_starts'] = dips_mask

    # detect artefacts and store in data properties
    data_properties['artefacts'] = {}
    mask_slinger_artefact(signal, data_properties)
    mask_calibration_artefacts(signal, data_properties)
    mask_flush_artefacts(signal, data_properties)
    mask_block_artefacts(signal, data_properties)

    solve_artefact_priority(data_properties, settings.ARTEFACT_PRIORITY)

    print(f"Detected artefacts:/n {data_properties['artefacts'].keys()}" )
    return data_properties
