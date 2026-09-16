"""
Respiratory Signal Processing & Benchmarking Pipeline

This script implements an end-to-end data processing pipeline for benchmarking 
an inkjet-printed Melanin-ChCl humidity sensor against clinical gold-standard 
respiratory sensors (thermistor & flow meter).

Key Functionalities:
1. Multi-Format Data Ingestion: Loads raw impedance CSVs (MFIA) and clinical EDF recordings.
2. Signal Conditioning: Applies low-pass filtering and a custom robust high-pass 
   drift-removal algorithm (downsampling-based) to eliminate baseline wander.
3. Automated Event Classification: Evaluates threshold crossings within defined time segments 
   to identify Apnea vs. Breathing events.
4. Feature Extraction: Resamples signals to a unified frequency (100 Hz) and extracts 
   real-time respiration rates (BPM) using peak detection via NeuroKit2.
5. Statistical Benchmarking: Performs window-averaged Bland-Altman analysis to evaluate 
   sensor bias and limits of agreement against medical reference devices.
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample_poly
import numpy as np
import matplotlib.cm as cm
import pyedflib
from scipy.signal import find_peaks

#def bland_altman_plot_windowed(bpm1, bpm2, sampling_rate, window_sec=20, title="Bland–Altman Plot (Windowed BPM)"):
def bland_altman_plot_windowed(bpm1, bpm2, sampling_rate, window_sec=20, title="Bland–Altman Plot (Windowed BPM)", method1='Method 1', method2='Method 2'):
    """
    Bland–Altman analysis on window-averaged BPM signals.

    Parameters
    ----------
    bpm1, bpm2 : array-like
        BPM time series from two methods
    sampling_rate : float
        Sampling rate of BPM signals (Hz)
    window_sec : float
        Window length in seconds
    title : str
        Plot title

    Returns
    -------
    stats : dict
        bias, sd_diff, loa_upper, loa_lower
    """

    bpm1 = np.asarray(bpm1)
    bpm2 = np.asarray(bpm2)

    assert bpm1.shape == bpm2.shape, "BPM vectors must have same length"

    window_samples = int(window_sec * sampling_rate)
    n_windows = len(bpm1) // window_samples

    # Truncate to full windows
    bpm1 = bpm1[:n_windows * window_samples]
    bpm2 = bpm2[:n_windows * window_samples]

    # Reshape: (windows, samples_per_window)
    bpm1_w = bpm1.reshape(n_windows, window_samples)
    bpm2_w = bpm2.reshape(n_windows, window_samples)

    # Window averages
    bpm1_mean = bpm1_w.mean(axis=1)
    bpm2_mean = bpm2_w.mean(axis=1)

    # Bland–Altman
    mean = (bpm1_mean + bpm2_mean) / 2
    diff = bpm1_mean - bpm2_mean

    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(mean, diff, alpha=0.6)
    plt.axhline(bias, color="red", linestyle="--", label=f"Bias = {bias:.2f}")
    plt.axhline(loa_upper, color="gray", linestyle="--", label="Upper LoA")
    plt.axhline(loa_lower, color="gray", linestyle="--", label="Lower LoA")

    plt.xlabel("Mean BPM (windowed)", fontsize=15, color='blue')
    #plt.ylabel("Difference BPM (Method 1 − Method 2)")
    plt.ylabel("Difference BPM ({} − {})".format(method1, method2), fontsize=15, color='blue')
    plt.title(title, fontsize=16, color='darkred', fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "bias": bias,
        "sd_diff": sd,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
        "n_windows": n_windows,
        "window_sec": window_sec
    }

def bland_altman_plot(bpm1, bpm2, title="Bland–Altman Plot"):
    bpm1 = np.asarray(bpm1)
    bpm2 = np.asarray(bpm2)

    assert bpm1.shape == bpm2.shape, "Inputs must have same shape"

    mean = (bpm1 + bpm2) / 2
    diff = bpm1 - bpm2

    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)

    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    plt.figure(figsize=(7,5))
    plt.scatter(mean, diff, alpha=0.4)
    plt.axhline(bias, color='red', linestyle='--', label=f'Bias = {bias:.2f}')
    plt.axhline(loa_upper, color='gray', linestyle='--', label='Upper LoA')
    plt.axhline(loa_lower, color='gray', linestyle='--', label='Lower LoA')

    plt.xlabel("Mean BPM")
    plt.ylabel("Difference BPM (Method 1 − Method 2)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "bias": bias,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
        "sd_diff": sd
    }

def analyze_rsp(data, which_sensor, sampling_rate=100, show_plots=True):
    """
    Analyze respiration signal using NeuroKit2.

    Parameters
    ----------
    data : pandas.Series or numpy.ndarray
        Respiration signal (RSP)
    sampling_rate : int or float
        Sampling rate in Hz
    show_plots : bool
        Whether to display plots

    Returns
    -------
    rsp_rate : pandas.Series
        Breaths per minute
    """

    import neurokit2 as nk
    import pandas as pd
    import matplotlib.pyplot as plt

    # Ensure pandas Series
    if not isinstance(data, pd.Series):
        data = pd.Series(data, name="RSP")
    else:
        data = data.rename("RSP")

    # Clean signal
    cleaned = nk.rsp_clean(data, sampling_rate=sampling_rate)

    # Extract and fix peaks
    _, peaks_dict = nk.rsp_peaks(cleaned)
    info = nk.rsp_fixpeaks(peaks_dict)

    # Plot raw + clean
    if show_plots:
        nk.signal_plot(
            pd.DataFrame({
                "RSP_Raw": data,
                "RSP_Clean": cleaned
            }),
            sampling_rate=sampling_rate,
            subplots=True
        )

        # Plot peaks (before / after fix)
        nk.events_plot(peaks_dict["RSP_Peaks"], cleaned)
        nk.events_plot(info["RSP_Peaks"], cleaned)

    # Respiration rate (using FIXED peaks)
    rsp_rate = nk.rsp_rate(
        cleaned,
        info["RSP_Peaks"],
        sampling_rate=sampling_rate
    )

    # Plot rate
    if show_plots:
        nk.signal_plot(rsp_rate, sampling_rate=sampling_rate)
        plt.ylabel("Breaths Per Minute", fontsize=14, color='blue')
        plt.xlabel("Time", fontsize=14, color='blue')
        plt.title('BPM for RSP with {} sensor'.format(which_sensor), fontsize=18, color='darkred', fontweight='bold')
        plt.show()

    return rsp_rate

def check_sampling_uniformity(time, plot=True, verbose=True):
    """
    time : 1D array-like of time stamps (seconds)
    Returns stats dict. If plot=True will show two plots: dt vs sample index, and histogram of dt.
    """
    t = np.asarray(time)
    if t.ndim != 1:
        raise ValueError("time must be 1D")

    # ensure monotonic
    diffs = np.diff(t)
    if np.any(diffs <= 0):
        # identify non-monotonic indices
        bad_idxs = np.where(diffs <= 0)[0]
        raise ValueError(f"Non-monotonic or duplicate time stamps found at diffs indices: {bad_idxs[:10]}")

    dt = diffs
    n = len(dt)
    mean_dt = np.mean(dt)
    median_dt = np.median(dt)
    std_dt = np.std(dt)
    min_dt = np.min(dt)
    max_dt = np.max(dt)
    cv = std_dt / mean_dt if mean_dt != 0 else np.nan
    pct_1 = np.percentile(dt, 1)
    pct_5 = np.percentile(dt, 5)
    pct_95 = np.percentile(dt, 95)
    pct_99 = np.percentile(dt, 99)

    # detect large gaps / missing samples (heuristic: gaps > 3 * median_dt)
    gap_threshold = 3 * median_dt
    large_gaps_idx = np.where(dt > gap_threshold)[0]
    n_large_gaps = len(large_gaps_idx)

    stats = {
        'n_intervals': n,
        'mean_dt': mean_dt,
        'median_dt': median_dt,
        'std_dt': std_dt,
        'cv': cv,
        'min_dt': min_dt,
        'max_dt': max_dt,
        'pct_1': pct_1,
        'pct_5': pct_5,
        'pct_95': pct_95,
        'pct_99': pct_99,
        'n_large_gaps': n_large_gaps,
        'large_gaps_indices_sample': large_gaps_idx[:20],
        'suggested_fs_from_mean': 1.0/mean_dt if mean_dt>0 else np.nan,
        'suggested_fs_from_median': 1.0/median_dt if median_dt>0 else np.nan,
        'gap_threshold_seconds': gap_threshold,
    }

    if verbose:
        print("Sampling uniformity summary:")
        print(f"  intervals: {n}")
        print(f"  mean dt: {mean_dt:.6e} s    median dt: {median_dt:.6e} s")
        print(f"  std dt: {std_dt:.6e} s    CV: {cv:.3e}")
        print(f"  min dt: {min_dt:.6e} s    max dt: {max_dt:.6e} s")
        print(f"  pct1/pct5/pct95/pct99: {pct_1:.3e} / {pct_5:.3e} / {pct_95:.3e} / {pct_99:.3e}")
        print(f"  large gaps (>3*median_dt): {n_large_gaps} (threshold {gap_threshold:.3e} s)")
        print(f"  suggested fs (1/mean_dt): {stats['suggested_fs_from_mean']:.3f} Hz")
        print(f"  suggested fs (1/median_dt): {stats['suggested_fs_from_median']:.3f} Hz")

    if plot:
        fig, ax = plt.subplots(2,1, figsize=(10,6), constrained_layout=True)
        ax[0].plot(dt, marker='.', linestyle='-', markersize=3)
        ax[0].axhline(mean_dt, color='C1', linestyle='--', label=f"mean dt={mean_dt:.3e}s")
        ax[0].axhline(median_dt, color='C2', linestyle=':', label=f"median dt={median_dt:.3e}s")
        ax[0].axhline(gap_threshold, color='r', linestyle='-.', label=f"gap threshold ({3:.0f}*med)")
        ax[0].set_ylabel("dt (s)")
        ax[0].set_title("dt between consecutive samples")
        ax[0].legend(fontsize='small')

        ax[1].hist(dt, bins=200)
        ax[1].set_xlabel("dt (s)")
        ax[1].set_title("Histogram of dt")
        plt.show()

    return stats


def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data)
    return filtered


def read_edf_file(file_path):
    f = pyedflib.EdfReader(file_path)
    channel_names = f.getSignalLabels()
    channel_freq = f.getSampleFrequencies()[0]
    recording = np.zeros((len(channel_names), f.getNSamples()[0]))
    for i in range(len(channel_names)):
        recording[i, :] = f.readSignal(i)
    f.close()
    return recording, channel_names, channel_freq


def extract_time_segment(recording, channel_freq, start_segment, end_segment):
    start_sample = int(start_segment * channel_freq)
    end_sample = int(end_segment * channel_freq)
    time_segment = recording[:, start_sample:end_sample]
    time_segment_time = np.linspace(start_segment, end_segment, time_segment.shape[1])
    return time_segment, time_segment_time

def highpass_filter_finite_difference(data, dt):
    """
    Applies a high-pass filter using the finite difference method.

    Args:
        data (np.ndarray): The signal data.
        dt (float): The time step between data points.

    Returns:
        np.ndarray: The filtered signal.
    """
    # The first difference approximates the derivative (high-pass filter)
    # The result is shorter by one sample
    y = np.diff(data, prepend=data[0]) / dt
    return y

def robust_highpass_remove_drift(data, cutoff, fs, order=4, target_fs=200.0):
    """
    Robust high-pass that avoids numerical issues when cutoff is extremely small
    relative to the sampling freq by downsampling first and subtracting the low-frequency drift.
    Args:
        data (np.ndarray): 1D signal.
        cutoff (float): desired cutoff frequency in Hz.
        fs (float): sampling frequency in Hz.
        order (int): Butterworth filter order.
        target_fs (float): target sampling freq after downsampling (Hz).
                           Use a value sufficient to represent the frequencies of interest,
                           e.g. 100-500 Hz. Only used if downsampling is triggered.
    Returns:
        np.ndarray: filtered signal (drift removed).
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be > 0")

    nyq = 0.5 * fs
    norm_cut = cutoff / nyq

    # Safety check for direct filtering
    if 0 < norm_cut < 1 and norm_cut >= 1e-5:
        # If normalized cutoff not extremely small, do direct filtfilt
        b, a = butter(order, norm_cut, btype='high', analog=False)
        return filtfilt(b, a, data)
    else:
        # Downsample strategy
        # choose integer downsample factor so new_fs ~ target_fs
        factor = max(1, int(round(fs / target_fs)))
        if factor == 1:
            # can't/shouldn't downsample, fallback to more tolerant cutoff check
            norm_cut = max(norm_cut, 1e-8)
            b, a = butter(order, norm_cut, btype='high', analog=False)
            return filtfilt(b, a, data)

        # downsample via resample_poly (good anti-aliasing)
        data_ds = resample_poly(data, up=1, down=factor)
        new_fs = fs / factor
        new_nyq = 0.5 * new_fs
        new_norm = min(max(cutoff / new_nyq, 1e-8), 0.9999)

        b, a = butter(order, new_norm, btype='high', analog=False)
        ds_highpassed = filtfilt(b, a, data_ds)

        # low-frequency component on downsampled signal = data_ds - highpassed_ds
        ds_low = data_ds - ds_highpassed

        # upsample drift back to original timebase (interpolation)
        t = np.arange(len(data)) / fs
        t_ds = np.arange(len(data_ds)) / new_fs
        drift_up = np.interp(t, t_ds, ds_low)

        # subtract drift from original data
        filtered = data - drift_up
        return filtered

def detect_threshold_crossings(time, signal, segments, threshold):
    """
    Detect if signal crosses threshold within given time segments.

    Parameters
    ----------
    time : np.ndarray
        Time vector (same length as signal).
    signal : np.ndarray
        Signal vector.
    segments : list of tuples
        Each tuple is (t_start, t_end).
    threshold : float
        Threshold value to check against.

    Returns
    -------
    results : list of int
        1 if threshold is crossed in segment, else 0.
    """
    results = []
    for t_start, t_end in segments:
        mask = (time >= t_start) & (time <= t_end)
        segment_sig = signal[mask]

        if len(segment_sig) == 0:
            results.append(0)  # no data in segment
            continue

        # בדיקה אם יש לפחות נקודה שעוברת את הסף
        if np.any(segment_sig > threshold):
            results.append(1)
        else:
            results.append(0)
    return results

def main():
    ####   Plot G3 12.6.2025 ####
    # ----------- First Plot: MFIA Data ------------
    csv_path = "Directory_path\\Data_file_Parralel_measurements_KIT_and_2_commercial_sensors_data_20250930_122848.csv"
    df = pd.read_csv(csv_path)
    df['AbsZ'] = pd.to_numeric(df['AbsZ'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    #print("df['time']", df['time'])
    df.dropna(subset=['AbsZ', 'time'], inplace=True)

    dt = np.diff(df['time'])
    avg_dt = np.mean(dt)
    fs = 1 / avg_dt
    print('MFIA recording (KIT signal), fs ==>', fs)
    cutoff_freq = 20
    df['AbsZ_filtered'] = lowpass_filter(df['AbsZ'], cutoff=cutoff_freq, fs=fs)
    # Check if sampling frequency is uniform
    #stats = check_sampling_uniformity(df['time'].values, plot=True)
    #print(stats)

    relevant_times_mfia = np.array([36.086, 42.639, 49.276, 56.102, 62.643, 69.265, 75.895, 82.507, 89.275, 95.864, 102.37, 109.04, 115.804, 122.461, 129.051, 135.92, 142.641, 149.228,
                                155.846, 162.564, 169.387, 183.569], dtype=np.float64)
    relevant_times_mfia = relevant_times_mfia - 36.086 + 1.4
    annotations_mfia = [#"Mark Start 1 with stop breathing",
                        "Mark Start 2 with quick breathing",
                        "Mark Start 3 with stop breathing",
                        "Normal breathing mouth close",
                        "Normal breathing mouth open",
                        "Apnea (Stop breathing)",
                        "Quick breathing mouth close",
                        "Quick breathing mouth open",
                        "Apnea (Stop breathing) ",
                        "Normal breathing mouth close",
                        "Normal breathing mouth open",
                        "Apnea (Stop breathing)",
                        "Normal breathing mouth close",
                        "Normal breathing mouth open",
                        "Apnea (Stop breathing)",
                        "Hypopnea (shallow breathing) mouth close",
                        "Hypopnea (shallow breathing) mouth close",
                        "Normal breathing mouth close",
                        "Normal breathing mouth open",
                        "Apnea (Stop breathing)",
                        "Normal breathing mouth close",
                        "Normal breathing mouth open",
                        "End"]
    relevant_annotations_mfia = np.array(annotations_mfia, dtype='<U40')
    print('relevant_annotations_mfia ', relevant_annotations_mfia)

    # ----------- Second Plot: EDF Airflow Sensor Data ------------
    edf_path = r"C:\Directory_path\Data_file_Parralel_measurements_KIT_and_2_commercial_sensors_data_20250930.edf"

    recording, channel_names, channel_freq = read_edf_file(edf_path)
    f = pyedflib.EdfReader(edf_path)
    annotations = f.readAnnotations()
    print("annotations  ", annotations)

    start_annotations = 'Manual Event: "Mark Start 1 with stop br'
    end_annotations = 'End Of Data: BGX-7B65'
    start_index = int(np.where(annotations[2] == start_annotations)[0]) + 1
    end_index = int(np.where(annotations[2] == end_annotations)[0])

    relevant_annotations_edf = annotations[2][start_index:end_index]
    relevant_times_edf = annotations[0][start_index:end_index]
    print("relevant_times_edf", relevant_times_edf)
    start_segment, end_segment = relevant_times_edf[0], relevant_times_edf[-1]
    time_segment, time_segment_time = extract_time_segment(recording, channel_freq, start_segment, end_segment)

    # --- High-Pass Filter Method (Finite Difference) ---
    signal = df['AbsZ_filtered']
    detrended_signal_filter_fd = highpass_filter_finite_difference(signal, df['time'][1]-df['time'][0])

    print('MFIA fs, check again ==>', fs)
    filtered_signal = -1 * robust_highpass_remove_drift(signal, cutoff=0.16, fs=fs, order=2, target_fs=50)

    # Detect Apnea
    # מקטעים לבדיקה
    segments = [(5.153, 11.79, "Quick breath"), (11.79, 18.616, "Possible Apnea"), (18.616, 25, "Breathing"), (25.16, 31.7, "Breathing"), (31.779, 38.409, "Possible Apnea"),
                (38.409, 45.021, "Quick breath"), (45.1, 51.7, "Quick breath"), (51.789, 58.378, "Possible Apnea"), (58.378, 64.884, "Breathing"), (64.91, 71.554, "Breathing"),
                (71.554, 78.318, "Possible Apnea"), (78.318, 84.975, "Breathing"), (85, 91.5, "Breathing"), (91.565, 98.434, "Possible Apnea"), (98.434, 105.155, "Shallow breath"),
                (105.155, 111.742, "Shallow breath"), (111.742, 118.36, "Breathing"), (118.36, 125.078, "Breathing"), (125.1, 131.85, "Possible Apnea"),
                (131.901, 146.083, "Breathing"), (146.083, 153, "Breathing")]

    # Detect Apnea KIT sensor
    threshold_KIT = 55000
    results_KIT = detect_threshold_crossings(df['time'], filtered_signal, [(s[0], s[1]) for s in segments], threshold_KIT)

    print('###############################')
    print('Cross threshold for KIT signal:')
    print('###############################')
    rows = []
    for (t_start, t_end, desc), res in zip(segments, results_KIT    ):
        status = 'Breathing detected' if res == 1 else 'Apnea (no crossing)'
        print(f"Segment {t_start}-{t_end}s ({desc}): {status}")
        rows.append({"Description": desc, "Breathing": res})

    # המרה ל-DataFrame
    BreathDetectedRows = pd.DataFrame(rows)

    # שמירה ל-CSV
    #append_or_create_csv(BreathDetectedRows, "segments_results_KIT_sensor.csv")

    # Detect Apnea commercial sensor
    threshold_commercial = 70
    # יצירת רשימה חדשה עם הערכים המתוקנים
    delta = 34.686
    segments_shifted = [(start + delta, end + delta, desc) for (start, end, desc) in segments]

    print(segments_shifted)
    results_commercial = detect_threshold_crossings(time_segment_time, time_segment[0, :], [(s[0], s[1]) for s in segments_shifted], threshold_commercial)

    print('######################################')
    print('Cross threshold for commercial signal:')
    print('######################################')
    rows2 = []
    for (t_start, t_end, desc), res in zip(segments_shifted, results_commercial    ):
        status = 'Breathing detected' if res == 1 else 'Apnea (no crossing)'
        print(f"Segment {t_start}-{t_end}s ({desc}): {status}")
        rows2.append({"Description": desc, "Breathing": res})

    # המרה ל-DataFrame
    BreathDetectedRows2 = pd.DataFrame(rows2)

    # ----------- Combined Plot ------------
    #fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    ##fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    num_channels = time_segment.shape[0]
    num_subplots = num_channels + 1
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 4 * num_subplots), sharex=False)

    # Top: MFIA data
    ax = axes[num_channels]
    ax.plot(df['time'], df['AbsZ_filtered'], label='|Z| filtered (20 Hz)', linewidth=2)
    ax.plot(df['time'], filtered_signal, label='Remove drift HPF', linewidth=2)
    print(f'time mfia filtered ====>', df['time'])  ########
    print(f'mfia filtered breath signal ====>', filtered_signal)  ########
    print('type filtered signal', type(filtered_signal))
    #ax1.plot(df['time'], detrended_signal_filter_fd, label='Detrended Signal (Finite Difference)', linewidth=2)
    colors1 = cm.hsv(np.linspace(0, 1, len(relevant_times_mfia)))
    for time, annotation, color in zip(relevant_times_mfia, relevant_annotations_mfia, colors1):
        print('added annotation to legend', annotation)
        ax.axvline(x=time, color=color, linestyle='--', label=annotation)
    ax.set_title("KIT humidity breath sensor (Impedance vs Time)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("AbsZ (Ohm)")
    ax.grid(True)
    ax.set_xlim(left=relevant_times_mfia[0], right=relevant_times_mfia[-1])
    handles, labels = ax.get_legend_handles_labels()
    print('handles debug ',handles)
    print('labels debug ', labels)
    labels[0] = '|Z| KIT sensor \ Voltage Commerical sensor'
    by_label = dict(zip(labels, handles))
    print('by_label debug', by_label)
    # יצירת זוגות (לשימור סדר + כפילויות)
    pairs = list(zip(handles, labels))
    #ax1.legend(by_label.values(), by_label.keys(), fontsize='small')

    # Top: MFIA data with filter - derivation
    #ax2.plot(df['time'], detrended_signal_filter_fd, label='Detrended Signal (Finite Difference)', linewidth=2)
    #colors2 = cm.hsv(np.linspace(0, 1, len(relevant_times_mfia)))
    #for time, annotation, color in zip(relevant_times_mfia, relevant_annotations_mfia, colors2):
    #    ax2.axvline(x=time, color=color, linestyle='--', label=annotation)
    #ax2.set_title("Detrended Signal (Finite Difference)")
    #ax2.set_xlabel("Time (s)")
    #ax2.set_ylabel("Derivative of AbsZ (Ohm)")
    #ax2.grid(True)
    #ax2.set_xlim(left=relevant_times_mfia[0], right=relevant_times_mfia[-1])

    for i in range(1, num_channels):
        axes[i].sharex(axes[0])

    # Bottom: Airflow sensor (EDF)
    print('number of channels',num_channels)
    for i in range(num_channels):
        print('i==>', i)
        ax = axes[i]
        ax.plot(time_segment_time, time_segment[i, :], label=channel_names[i])
        print(f'time_edf_ch {channel_names[i]}====>', time_segment_time) ########
        print()

        # קווים אנכיים עם אנוטציות
        colors3 = cm.hsv(np.linspace(0, 1, len(relevant_times_edf)))
        for time, annotation, color in zip(relevant_times_edf, relevant_annotations_edf, colors3):
            ax.axvline(x=time, color=color, linestyle='--', label=annotation)

        print('channel_names[i] ', channel_names[i])
        #ax.set_title(f"Commercial Airflow Sensor - Channel {channel_names[i]}")
        match channel_names[i]:
            case "EEG1_FC":
                ax.set_title(f"Commercial Airflow Sensor (Temperature)")
            case "EEG4_FC":
                ax.set_title(f"Commercial Airflow Sensor (Flow)")
            case _:
                ax.set_title(f"Commercial Airflow Sensor - Channel {channel_names[i]}")

        ax.set_ylabel("Voltage")
        ax.grid(True)
        ax.set_xlim(left=relevant_times_edf[0], right=relevant_times_edf[-1])

        # כדי למנוע כפילות בלג'נד
        handles2, labels2 = ax.get_legend_handles_labels()
        labels[0] = '|Z| KIT sensor \ Voltage Commerical sensor'
        by_label2 = dict(zip(labels2, handles2))
        #ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    plt.subplots_adjust(bottom=0.18)  # יותר מקום מתחת
    # יצירת לג'נד משותף לכל הפיגור
    print('labels=> ',by_label.keys())
    #fig.legend(by_label.values(), by_label.keys(),
    fig.legend([p[0] for p in pairs],  # כל ה-handles
                [p[1] for p in pairs],  # כל ה-labels (גם כפולים)
               loc="center",  # אפשר לשנות ל 'upper center' / 'right' / 'left'
               bbox_to_anchor=(0.5, 0.07),
               fontsize='small',
               ncol=4)  # שליטה על מספר העמודות

    #plt.tight_layout()
    plt.show()
    #### Breath rate #####
    # KIT sensor, NumPy → Series
    dt = df['time'].diff().dropna() # calculate time diffrence delta t
    fs = 1.0 / dt.mean()  # mean sampling rate (Hz)
    print('fs KIT sensor, measured with MFIA: ', fs)
    # Check if rate is really stable
    print("dt mean:", dt.mean())
    print("dt std :", dt.std())
    # Calculate mode (more stable then average)
    dt_mode = dt.round(9).mode()[0]
    fs_mode = 1.0 / dt_mode
    print('fs_mode: ', fs_mode)
    # Downsampling
    target_fs = 100  # או 50
    factor = int(fs / target_fs)

    signal_ds_KIT = filtered_signal[::factor]
    time_ds = df['time'].iloc[::factor].reset_index(drop=True)
    # Check
    new_fs = 1.0 / np.diff(time_ds).mean()
    print('New fs after Downsampling' ,new_fs)

    # bpm for KIT sensor data
    N=14749
    rsp = signal_ds_KIT[:N]
    bpm = analyze_rsp(rsp, which_sensor='Melanin-ChCl, ', sampling_rate=100)
    print('bpm KIT sensor ==>', bpm)
    print('Mean bpm KIT sensor ==>', bpm.mean())

    ######################################
    # bpm for flow and temperature sensors
    ######################################
    # Temperature sensor - bpm
    ######################################
    print('chanel 0 shape ', time_segment[0, :].shape)
    print('chanel 0 shape ',time_segment[0, :].dtype)
    print('time shape ', time_segment_time.shape)
    print('time stype ',time_segment_time.dtype)
    print('Check sampling frequency')
    dt = np.diff(time_segment_time) # calculate time diffrence delta t
    fs = 1.0 / dt.mean()  # mean sampling rate (Hz)
    print('fs KIT sensor, measured with MFIA: ', fs)
    # Check if rate is really stable
    print("dt mean:", dt.mean())
    print("dt std :", dt.std())
    # Calculate mode (more stable then average)
    dt_mode = pd.Series(dt).round(9).mode()[0]
    fs_mode = 1.0 / dt_mode
    print('fs_mode: ', fs_mode)
    # Downsampling (from250 to 100)
    # New time axis
    fs_target = 100  # Hz
    t_start = time_segment_time[0]
    t_end = time_segment_time[-1]

    time_ds = np.arange(t_start, t_end, 1 / fs_target)
    signal_ds_temp = np.interp(time_ds, time_segment_time, time_segment[0, :]) # interpolation of the signal
    # Sanity check
    dt_ds = np.diff(time_ds)
    print("fs new (~100?):", 1 / dt_ds.mean())
    bpm_ch0 = analyze_rsp(signal_ds_temp, which_sensor='Temperature', sampling_rate=100)
    print('bpm Temperature sensor ==>', bpm_ch0)
    print('Mean Temperature sensor ==>', bpm_ch0.mean())
    ######################################
    # Flow sensor - bpm
    ######################################
    print('chanel 1 shape ', time_segment[1, :].shape)
    print('chanel 1 shape ',time_segment[1, :].dtype)
    print('Check sampling frequency')
    dt = np.diff(time_segment_time) # calculate time diffrence delta t
    fs = 1.0 / dt.mean()  # mean sampling rate (Hz)
    print('fs KIT sensor, measured with MFIA: ', fs)
    # Check if rate is really stable
    print("dt mean:", dt.mean())
    print("dt std :", dt.std())
    # Calculate mode (more stable then average)
    dt_mode = pd.Series(dt).round(9).mode()[0]
    fs_mode = 1.0 / dt_mode
    print('fs_mode: ', fs_mode)
    # Downsampling (from250 to 100)
    # New time axis
    fs_target = 100  # Hz
    t_start = time_segment_time[0]
    t_end = time_segment_time[-1]

    time_ds = np.arange(t_start, t_end, 1 / fs_target)
    signal_ds_flow = np.interp(time_ds, time_segment_time, time_segment[1, :]) # interpolation of the signal
    # Sanity check
    dt_ds = np.diff(time_ds)
    print("fs new (~100?):", 1 / dt_ds.mean())
    bpm_ch1 = analyze_rsp(signal_ds_flow, which_sensor='Flow', sampling_rate=100)
    print('bpm Flow sensor ==>', bpm_ch1)
    print('Mean Flow sensor ==>', bpm_ch1.mean())
    print('Size of bpm vec: ')
    print('     KIT: ', bpm.shape)
    print('     Temperature: ', bpm_ch0.shape)
    print('     Flow: ', bpm_ch1.shape)
    print('Size of signal vec: ')
    print('     KIT: ', signal_ds_KIT.shape)
    print('     Temperature: ', signal_ds_temp.shape)
    print('     Flow: ', signal_ds_flow.shape)

    ########################################
    # Bland-Altman plot to compare methods #
    ########################################
    bpm_method1 = bpm_ch0 # Temperature
    bpm_method2 = bpm_ch1 # Flow
    # stats = bland_altman_plot(bpm_method1, bpm_method2)
    stats = bland_altman_plot_windowed(bpm_method1, bpm_method2, sampling_rate=100, window_sec=10, method1='Temperature', method2='Flow')
    print(stats)
    N = min(len(bpm), len(bpm_ch0))
    print('N', N)
    bpm_KIT = bpm[:N]
    bpm_temperature = bpm_ch0[:N]

    #for w in [5, 8, 10, 15]:
    #    stats = bland_altman_plot_windowed(bpm_KIT, bpm_temperature, sampling_rate=100, window_sec=w, title=f"Bland–Altman (window={w}s)", method1='Humidity', method2='Temperature')
    #    print(w, stats)
    #stats = bland_altman_plot_windowed(bpm_KIT, bpm_temperature, sampling_rate=100, window_sec=10, method1='Melanin Ch-Cl', method2='Thermistor')
    stats = bland_altman_plot_windowed(bpm_KIT, bpm_temperature, sampling_rate=100, window_sec=10,
                                       method1='Humidity', method2='Temperature')
    print(stats)


if __name__ == "__main__":
    main()
