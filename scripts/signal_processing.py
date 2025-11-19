"""
Signal processing module
"""
import datetime as dtime

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt

import scripts.settings as settings

def convert_to_datetime(st: str) -> dtime.datetime:
    """
    Convert string to datetime object.
    """
    return dtime.datetime.strptime(st[:23] + '000', '%Y-%m-%d %H:%M:%S.%f')

def build_signal_and_time(db, range_to_plot) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """
    Build full signal and time vector from database within specified range.
    """
    # sampling interval in seconds (from settings)
    dt = settings.SAMPLING_DT

    # ensure Time column contains datetime objects
    db['Time'] = db['Time'].map(convert_to_datetime)

    # starting time for building the continuous time vector
    starttime = db['Time'][0]

    # containers for assembled signal and time vectors
    signal = []
    time = []

    # separate arrays for the CCO timestamps and values for the requested range
    time_cco = db['Time'][range_to_plot]
    cco = db['CCO'][range_to_plot]

    # iterate over the requested segment indices and append wave samples
    for idx in range_to_plot:
        n = len(db['Wave'][idx])  # number of samples in this wave segment

        # determine the start time for this segment:
        # if time already has elements, start immediately after the last appended sample,
        # otherwise use the original starttime from the DB
        sect_time = time[-1] + dtime.timedelta(seconds=dt) if time else starttime

        # update the DB Time for this index to the computed segment start
        db.loc[idx, 'Time'] = sect_time

        # build a datetime vector for each sample in the segment with spacing dt
        datetime_vector = [sect_time + dtime.timedelta(seconds=i * dt) for i in range(n)]

        # extend the master time and signal lists
        time.extend(datetime_vector)
        signal.extend(db['Wave'][idx])

    # ensure proper numpy arrays are returned
    time = np.array(time)
    signal = np.array(signal)
    time_cco = np.array(db['Time'].iloc[range_to_plot])
    cco = np.array(cco)
    return time, signal, time_cco, cco

# def find_connected_segments(starts, ends) -> list[tuple[int, int]]:
#     """
#     Find connected segments from start and end indices.
#     """

#     # raise error if starts or ends are empty
#     if starts.size == 0 or ends.size == 0:
#         raise ValueError("Starts and ends must be non-empty arrays.")

#     # combine and sort start and end events
#     events = np.concatenate([starts, ends])
#     labels = np.concatenate([['start']*len(starts), ['end']*len(ends)])
#     order = np.argsort(events)
#     events = events[order]
#     labels = labels[order] #sort lables similarly

#     # ensure last event is a start to close any open segments
#     if labels[-1] != 'start':
#         max_idx = np.max(events)
#         events = np.append(events, max_idx + 1)
#         labels = np.append(labels, 'start')

#     # init containers
#     segments = []
#     active_starts = []
#     active_ends = []
#     segment_open = False


#     for idx, label in zip(events, labels):
#         # ===== Segment start ========================================================
#         if label == 'start':
#             if segment_open:
#                 active_starts.append(idx) # accumulate all starts within open segment
#             else:
#                 # ==== Segment Closure ===============================================
#                 if active_ends:
#                     thresh = settings.SAMPLING_FREQUENCY * 5 # max 5 seconds between points

#                     diff_ends = np.diff(active_ends)
#                     outlier_indices_end = np.where(diff_ends > thresh)[0]

#                     if len(outlier_indices_end) > 0:
#                         # remove all end indices after the last large gap
#                         last_keep = outlier_indices_end[0] + 1
#                         active_ends = active_ends[last_keep:]

#                     max_end = max(active_ends)
#                     active_ends = []
#                     try:
#                         # segment must be shorter than max artefact duration in seconds
#                         if max_end - min_start > (settings.SAMPLING_FREQUENCY *
#                                                   settings.MAX_ARTEFACT_DURATION):
#                             del min_start, max_end
#                         else: # if valid segment, store it as tuple in segments list
#                             segments.append((min_start, max_end))
#                             del min_start, max_end
#                     except NameError: # if min_start or max_end is not defined
#                         pass
#                     active_ends = []
#                     active_starts = []
#                 active_starts = [idx]
#                 segment_open = True

#         # ===== Segment end ==========================================================
#         elif label == 'end':  # end
#             active_ends.append(idx) # accumulate all ends within open segment

#             if segment_open:
#                 thresh = settings.SAMPLING_FREQUENCY * 5 # max 5 seconds between points

#                 diff_starts = np.diff(active_starts)
#                 outlier_indices_start = np.where(diff_starts > thresh)[0]

#                 if len(outlier_indices_start) > 0:
#                     # Remove all outlier indices at once
#                     active_starts = np.delete(active_starts, range(outlier_indices_start[-1]+1))

#                 min_start = min(active_starts)

#                 segment_open = False
#                 active_starts = []

#     return segments

def build_mask(indices, length) -> np.ndarray:
    """
    Return boolean mask of given length with True at indices.
    """
    # initialize boolean mask of given length (all False)
    mask = np.zeros(length, dtype=bool)

    # if no indices provided, return the all-False mask
    if indices is None:
        return mask

    # ensure indices are an integer numpy array
    indices = np.asarray(indices, dtype=int)

    # keep only indices within the valid range [0, length-1]
    valid_mask = (indices >= 0) & (indices < length)
    idx = indices[valid_mask]

    # mark valid positions as True (duplicates are harmless)
    if idx.size:
        mask[idx] = True

    return mask

def build_threshold_mask(y, threshold: list) -> np.ndarray:
    """
    Create a boolean mask where the absolute signal exceeds the threshold.
    """
    threshold_mask = np.logical_or(np.abs(y) < threshold[0], np.abs(y) > threshold[1])
    return threshold_mask

def calc_heart_rate(time, signal) -> tuple[float, np.ndarray]:
    """
    Calculate heart rate from peak indices.
    """

    # detect peaks in the signal using a minimal distance (in samples)
    # distance=int(0.4 / settings.sampling_dt) enforces at least 0.4 seconds between peaks
    peak_indices, _ = find_peaks(signal, distance=int(0.4 / settings.SAMPLING_DT))

    # if fewer than two peaks found, cannot compute heart rate -> return NaN
    if peak_indices.size < 2:
        heart_rates = np.nan
    else:
        # extract the datetime timestamps corresponding to detected peaks
        peak_times = time[peak_indices]

        # compute successive differences (RR intervals) as timedeltas
        # convert to milliseconds then to seconds (float)
        rr_intervals = np.diff(peak_times).astype('timedelta64[ms]').astype(float) / 1000.0

        # discard non-positive intervals (defensive)
        rr_intervals = rr_intervals[rr_intervals > 0]

        # if no valid intervals remain, return NaN
        if rr_intervals.size == 0:
            heart_rates = np.nan
        else:
            # instantaneous frequency in beats per second
            bps = 1 / rr_intervals  # bps

            # use median of instantaneous frequencies and convert to bpm
            heart_rates = float(np.median(bps)) * 60  # bpm

    # return computed heart rate (bpm or NaN) and the indices of detected peaks
    return heart_rates, peak_indices

def find_dips(signal) -> np.ndarray:
    """
    Find dips in the signal using peak prominences.
    """
    dip_indices, _ = find_peaks(-signal, distance=int(0.4 / settings.SAMPLING_DT))
    return dip_indices

def calculate_fft(y) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate single-sided FFT amplitude spectrum.
    Based on settings.sampling_dt
    """

    y = np.asarray(y, dtype=float)

    # compute median sampling interval in seconds (supports datetimes or numeric)
    dt = settings.SAMPLING_DT
    n = y.size
    y = y - np.mean(y)  # remove DC

    fft_y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, dt)

    # single-sided amplitude spectrum (scaled)
    amplitude = (2.0 / n) * np.abs(fft_y)
    return freqs, amplitude

def butterworth_filt(y,fc,order, btype: str) -> np.ndarray:
    """
    Butterworth Filter
    btype{'lowpass', 'highpass', 'bandpass', 'bandstop'}
    """
    fs = settings.SAMPLING_FREQUENCY
    b, a = butter(order, fc, fs=fs, btype=btype, analog=False)

    filtered_test = filtfilt(b, a, y)

    return filtered_test

def mask_artefact_windows(start_sys_mask, raw_artefact_mask, n_cycles=2) -> np.ndarray:
    """
    Mark artefact windows in slinger_mask by expanding around systolic start indices
    when outliers occur between consecutive systolic starts.
    Expands by n_cycles systolic starts on each side (2*n_cycles+1 cycles total).
    """
    artefact_mask = np.zeros(len(raw_artefact_mask), dtype=bool)

    if start_sys_mask is None:
        raise ValueError("start_sys_mask cannot be None.")

    start_sys_indices = np.where(start_sys_mask)[0]
    # need at least n_cycles before and n_cycles after + the current one
    if start_sys_indices.size < (n_cycles * 2 + 1):
        raise ValueError("Not enough systolic start indices to build artefact windows. " \
                         "Increase data length or decrease n_cycles.")

    n = len(artefact_mask)
    for i in range(len(start_sys_indices) - 1):
        # ensure there are n_cycles systolic starts before and after
        if i >= n_cycles and (i + n_cycles) < len(start_sys_indices):
            s_idx = start_sys_indices[i]
            e_idx = start_sys_indices[i + 1]
            # if any outlier falls between these two systolic starts, mark a wider window
            if raw_artefact_mask[s_idx:e_idx].any():
                # expand window by n_cycles systolic starts on each side
                artefact_start = start_sys_indices[i - n_cycles]
                artefact_end = start_sys_indices[i + n_cycles]
                # create artefact mask ensuring indices are within signal bounds
                artefact_start = max(0, int(artefact_start))
                artefact_end = min(n, int(artefact_end))
                artefact_mask[artefact_start:artefact_end] = True
    return artefact_mask

def mask_slinger_artefact(signal, data_properties) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Mask artefacts in the signal based on threshold.
    """
    # filter signal with highpass Butterworth filter, to enhance artefact features
    filt = np.abs(butterworth_filt(signal, fc=20, order=4, btype='highpass') ** 3)

    peaks = find_peaks(filt, distance=int(settings.DISTANCE_SLINGER / settings.SAMPLING_DT),
                       prominence=settings.PROMINENCE_SLINGER)[0]

    # exit if no peaks found
    if peaks.size == 0:
        data_properties['slinger_artefacts'] = None
        return data_properties

    # calculate deviations from mean peak value and identify outliers
    abs_peak = np.abs(filt[peaks])
    dev = np.abs(abs_peak - np.median(abs_peak))
    outliers = peaks[dev > 3*np.median(dev)]
    if outliers.size == 0:
        data_properties['slinger_artefacts'] = None
        return data_properties

    vals = filt[outliers]
    if vals.size == 0 or vals.max() < 0.4:
        print("No significant slinger artefacts detected.")
        return data_properties

    # make emtpy mask and fill in artefact points
    raw_mask = np.zeros(len(signal), dtype=bool)
    raw_mask[outliers] = True

    # make wider artefact windows around systolic starts (2 cycles on each side)
    slinger_mask = mask_artefact_windows(data_properties.get('systolic_starts'),
                                         raw_mask, n_cycles=2)

    data_properties['artefacts']['slinger_artefacts'] = slinger_mask

    return data_properties

def mask_calibration_artefacts(signal, data_properties) -> dict:
    """
    Mask calibration artefacts in the signal based on sudden drops.
    """
    # detect large negative drops in the signal
    calibration_indices = np.where((signal < settings.CALIBRATION_SIGNAL_THRESHOLD))[0]
    
    #make a list of indices for each segment
    valid_calibration_indices = []

    if calibration_indices.size == 0:
        valid_calibration_indices = []
        start_idxs = []
        end_idxs = []
    else:
        # find artefacts runs by splitting at gaps larger than 1 sample
        diffs = np.diff(calibration_indices)
        split_points = np.where(diffs > 1)[0]
        runs = np.split(calibration_indices, split_points + 1)
        valid_calibration_indices = []
        start_idxs = []
        end_idxs = []
        for run in runs:
            # accept runs that are at least 3 seconds long
            if run.size >= settings.SAMPLING_FREQUENCY * 3:
                valid_calibration_indices.extend(range(int(run[0]), int(run[-1]) + 1))
                start_idxs.append(int(run[0]))
                end_idxs.append(int(run[-1]))

    if len(valid_calibration_indices) == 0:
        print("No significant calibration artefacts detected.")
        return data_properties

    # mask calibration artefacts inbetween start and endpoints.
    calibration_mask = np.zeros(len(signal), dtype=bool)
    calibration_mask[valid_calibration_indices] = True

    # make wider artefact windows around systolic starts (2 cycles on each side)
    calibration_artefact_mask = mask_artefact_windows(data_properties.get('systolic_starts'),
                                                calibration_mask, n_cycles=2)
    data_properties['artefacts']['calibration_artefacts'] = calibration_artefact_mask

    return data_properties

def mask_flush_artefacts(signal, data_properties) -> dict:
    """
    Mask flush artefacts in the signal based on sudden spikes.
    """
    # detect large positive spikes in the signal
    flush_indices = np.where((signal > settings.FLUSH_SIGNAL_THRESHOLD))[0]

    # calculate rolling standard deviation with a window of 1 second
    window_size = int(settings.SAMPLING_FREQUENCY * 1)  # 1 second window

    # rolling standard deviation calculation
    rolling_std = np.array([
        np.std(signal[max(0, i - window_size // 2):min(len(signal), i + window_size // 2)])
        for i in range(len(signal))
    ])

    low_std_indices = np.where(rolling_std < settings.FLUSH_STD_THRESHOLD)[0]

    #make a list of indices for each segment
    valid_flush_indices = []

    if flush_indices.size == 0:
        valid_flush_indices = []
        start_idxs = []
        end_idxs = []
    else:
        # find artefacts runs by splitting at gaps larger than 1 sample
        diffs = np.diff(flush_indices)
        split_points = np.where(diffs > 1)[0]
        runs = np.split(flush_indices, split_points + 1)
        valid_flush_indices = []
        start_idxs = []
        end_idxs = []
        for run in runs:
            # accept runs that are at least 3 seconds long
            run_low_std = [idx for idx in run if idx in low_std_indices]
            if run.size >= settings.SAMPLING_FREQUENCY * 3 and len(run_low_std) / run.size >= 0.5:
                valid_flush_indices.extend(range(int(run[0]), int(run[-1]) + 1))
                start_idxs.append(int(run[0]))
                end_idxs.append(int(run[-1]))

    if len(valid_flush_indices) == 0:
        print("No significant flush artefacts detected.")
        return data_properties

    # mask calibration artefacts inbetween start and endpoints.
    flush_mask = np.zeros(len(signal), dtype=bool)
    flush_mask[valid_flush_indices] = True

    # make wider artefact windows around systolic starts (2 cycles on each side)
    flush_artefact_mask = mask_artefact_windows(data_properties.get('systolic_starts'),
                                                flush_mask, n_cycles=2)

    data_properties['artefacts']['flush_artefacts'] = flush_artefact_mask

    return data_properties

def mask_block_artefacts(signal, data_properties) -> dict:
    """
    Mask block artefacts in the signal based on standard deviation.
    """



    # calculate rolling standard deviation with a window of 1 second
    window_size = int(settings.SAMPLING_FREQUENCY * 1)  # 1 second window

    # rolling standard deviation calculation
    rolling_std = np.array([
        np.std(signal[max(0, i - window_size // 2):min(len(signal), i + window_size // 2)])
        for i in range(len(signal))
    ])

    outliers = np.where(rolling_std < settings.BLOCK_STD_THRESHOLD)[0]

 #  make emtpy mask and fill in artefact points
    raw_mask = np.zeros(len(signal), dtype=bool)
    raw_mask[outliers] = True

    # make wider artefact windows around systolic starts (2 cycles on each side)
    block_artefacts = mask_artefact_windows(data_properties.get('systolic_starts'),
                                         raw_mask, n_cycles=2)

    data_properties['artefacts']['block_artefacts'] = block_artefacts

    return data_properties

def solve_artefact_priority(data_properties, artefact_priority) -> dict:
    """
    Determine priority of artefact types.
    Lower number indicates higher priority.
    """

    artefact = data_properties['artefacts'].keys()

    # sort artefact types by priority
    sorted_artefacts = sorted(artefact, key=lambda x: artefact_priority.get(x, float('inf')))

    # delete lower priority artefacts in lower priority masks
    for artefact in sorted_artefacts:
        current_priority = artefact_priority.get(artefact, float('inf'))
        current_mask = data_properties['artefacts'][artefact]
        for other_artefact in sorted_artefacts:
            if other_artefact == artefact:
                continue
            other_priority = artefact_priority.get(other_artefact, float('inf'))
            if other_priority > current_priority:
                other_mask = data_properties['artefacts'][other_artefact]
                # remove current artefact points from lower priority masks
                other_mask[current_mask] = False
                data_properties['artefacts'][other_artefact] = other_mask

    return data_properties
