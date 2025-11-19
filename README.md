# Artefact_Detect

Tooling to load, process and visualise continuous arterial pressure signals, and automatically detect common artefacts (slinger, calibration, flush, block).

This is a small research codebase used in a technical medicine context. The goal is to go from raw recorded data (CSV → pickle) to:

- a continuous time/signal vector,
- derived features such as heart rate and FFT,
- boolean masks for different artefact types, and
- clear plots that highlight those artefacts on top of the raw waveform.

---

## 1. Quick start

1. Make sure you have Python 3.10+ installed.
2. Open a PowerShell in the project folder `Artefact_Detect`.
3. (Optional but recommended) create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the required Python packages:

   ```powershell
   pip install numpy scipy matplotlib seaborn pandas
   ```

5. Place your input data file(s) in the `data` folder (e.g. `decoded_anonymous_data_1.pkl`).
6. Run the main script:

   ```powershell
   python .\main.py
   ```

You should see a figure window with:

- the full arterial pressure signal in grey,
- markers for systolic peaks,
- coloured overlays for each detected artefact type,
- a secondary y‑axis showing the CCO signal.

---

## 2. Repository structure

- `main.py` — entry point; wires everything together, runs artefact detection, and produces the main plots.
- `artefact_detection.py` — high‑level logic for detecting artefacts and assembling a `data_properties` dictionary.
- `scripts/`
  - `settings.py` — configuration of sampling rate, thresholds and file paths.
  - `filer.py` — helpers to read CSVs and pickle databases.
  - `signal_processing.py` — core signal processing utilities (build continuous signal, FFT, masks, filters, artefact helpers).
  - `plot.py` — lightweight plotting helper (`Plotter` class) used by `main.py`.
- `data/` — expected location for input files (CSV or pickle).
- `output/` — expected location for generated databases and results.

You can treat `main.py` as the "demo" of how all pieces fit together.

---

## 3. Configuration (`scripts/settings.py`)

All important parameters are centralised in `scripts/settings.py`. The function `settings.init()` must be called once at the start of a run (this is done inside `main.py`). Key settings:

- **Measurement / sampling**
  - `SAMPLING_DT` — sampling interval in seconds (e.g. `0.008`).
  - `SAMPLING_FREQUENCY` — derived as `1 / SAMPLING_DT`.

- **File paths**
  - `OUTPUT_DIR` — where processed pickle databases or results are written.
  - `INPUT_DIR` — where input CSV / decoded pickle files live.
  - `DATABASE_DIR` — default path to a database pickle file.

- **Artefact thresholds**
  - `DISTANCE_SLINGER`, `PROMINENCE_SLINGER` — minimum spacing and prominence for slinger artefacts.
  - `CALIBRATION_STD_THRESHOLD`, `CALIBRATION_SIGNAL_THRESHOLD` — thresholds for calibration artefacts.
  - `FLUSH_STD_THRESHOLD`, `FLUSH_SIGNAL_THRESHOLD` — thresholds for flush artefacts.
  - `BLOCK_STD_THRESHOLD` — used to detect block artefacts.

- **Priority of artefacts**
  - `ARTEFACT_PRIORITY` — mapping from artefact name to priority rank.

Adjust these values to match the characteristics of your signal and measurement protocol before running analyses on new datasets.

---

## 4. Data flow and main pipeline

The standard analysis in `main.py` follows this sequence:

1. **Initialise settings**

   ```python
   import scripts.settings as settings
   settings.init()
   ```

2. **Load database** using `scripts.filer.load_database`:

   ```python
   from scripts.filer import load_database
   db = load_database(r'decoded_anonymous_data_1.pkl')
   ```

3. **Build continuous time and signal** using `build_signal_and_time`:

   ```python
   from scripts.signal_processing import build_signal_and_time
   time, signal, time_cco, cco = build_signal_and_time(db, range(0, 1000))
   ```

   - The original data is stored per segment; this function stitches segments together into a continuous NumPy `time` array (datetime) and a `signal` array.

4. **Run artefact detection** using `artefact_detection.artefact_detection`:

   ```python
   from artefact_detection import artefact_detection
   data_properties = artefact_detection(time, signal)
   ```

5. **Plot results** using the `Plotter` class from `scripts.plot`:

   ```python
   from scripts.plot import Plotter

   plot = Plotter([['main']], fig_title='Full Signal with CCO Overlay')
   plot.add_plot(time, signal, axis_key='main',
                 color='grey', linewidth=1, alpha=0.7,
                 xlabel='Time', ylabel='Signal Amplitude (a.u.)',
                 label='Raw Signal')

   plot.add_plot(time[data_properties['systolic_peaks']],
                 signal[data_properties['systolic_peaks']],
                 axis_key='main',
                 color='green', linestyle='None', marker='o',
                 label='Systolic Starts')

   for i, key in enumerate(data_properties['artefacts'].keys()):
       mask = data_properties['artefacts'][key]
       plot.add_plot(np.ma.masked_array(time, ~mask),
                     np.ma.masked_array(signal, ~mask),
                     axis_key='main', linewidth=2, alpha=0.8,
                     label=key.replace('_', ' ').title())

   plot.add_secondary_y_axes(original_axis_key='main', secondary_axis_key='cco', label='CCO (l/min)')
   plot.add_plot(time_cco, cco, axis_key='cco',
                 color='orange', linewidth=2, alpha=0.7,
                 label='CCO Signal')

   plot.show()
   ```

This is essentially what `main.py` already does; you can modify it to fit your own analysis.

---

## 5. Module overview

### `scripts/filer.py`
- **Purpose:** Central place for reading and writing data files so the rest of the code does not deal with file paths directly.
- **Key functions:**
   - `load_csv_as_dict(filename)`: reads a CSV from `settings.INPUT_DIR`, stores the DataFrame under its filename key in a dict, and saves that dict as a pickle in `settings.OUTPUT_DIR` (prefixed with `coded_`).
   - `load_database(filename)`: loads an existing pickle (e.g. `decoded_anonymous_data_1.pkl`) from `settings.INPUT_DIR` and returns the stored structure (usually a dict with a pandas DataFrame).
   - `save_pkl(db, filename)`: saves a database dict to `settings.OUTPUT_DIR` with a `decoded_` prefix, used when you have processed data you want to reuse later.

### `scripts/signal_processing.py`
- **Purpose:** All low‑level signal utilities live here so they can be reused independently of plotting or I/O.
- **Key responsibilities:**
   - Construct continuous `time` and `signal` arrays from segmented data (`build_signal_and_time`).
   - Build boolean masks from indices (`build_mask`) or thresholds (`build_threshold_mask`).
   - Compute derived metrics (heart rate via `calc_heart_rate`, FFT via `calculate_fft`).
   - Apply filters (e.g. `butterworth_filt`) and detect event indices (peaks, dips).
   - Implement artefact‑specific helpers such as `mask_slinger_artefact`, `mask_calibration_artefacts`, `mask_flush_artefacts`, `mask_block_artefacts` and `mask_artefact_windows`.

### `artefact_detection.py`
- **Purpose:** Glue module that combines signal‑processing steps into a single call used by `main.py`.
- **Core function:**
   - `artefact_detection(time, signal) -> dict`: orchestrates the full analysis pipeline:
      1. Detects systolic peaks and calculates an estimated heart rate (`calc_heart_rate`).
      2. Builds masks for peaks and systolic starts (`build_mask`, `find_dips`).
      3. Calculates FFT information (`calculate_fft`) and stores the dominant frequency.
      4. Runs all artefact maskers (slinger, calibration, flush, block) and stores the resulting boolean masks under `data_properties['artefacts']`.
      5. Resolves overlap between artefacts using `solve_artefact_priority` and the priorities defined in `settings.ARTEFACT_PRIORITY`.

### `scripts/plot.py`
- **Purpose:** Provide a small, opinionated wrapper around Matplotlib so plots in this project look consistent and are easy to extend.
- **Key component:**
   - `Plotter` class:
      - Constructor `Plotter(mosaic=None, fig_title='...', figsize=(10, 6))` creates a figure and axes (optionally with a mosaic layout) and configures Seaborn/Matplotlib style.
      - `add_plot(...)` adds lines with convenient styling arguments (`color`, `linestyle`, `linewidth`, `marker`, `label`, `alpha`, etc.).
      - `add_fill_between(...)` draws shaded regions, for example to highlight ranges.
      - `add_secondary_y_axes(...)` adds a second y‑axis that shares the same x‑axis, used for CCO in `main.py`.
      - `sync_x_axes(...)` ties multiple axes to the same x‑scale if you create more than one panel.
      - `set_yticks(...)` helps standardise tick placement across panels.
      - `show()` is a thin wrapper around `plt.show()`.

---

## 6. License and authorship

This code is currently used in a research/educational context. Add an explicit license (e.g. MIT, BSD, or similar) if you plan to share or publish the repository.

Original author: Ming Dao Caljé (247116), see headers in the source files.
