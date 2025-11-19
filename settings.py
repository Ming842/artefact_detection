from typing import Optional, Dict

"""
Settings module for artefact detection parameters and file paths.
"""

# Module-level defaults for globals used in init().
# Module-level defaults for globals used in init().
SAMPLING_DT: Optional[float] = None
SAMPLING_FREQUENCY: Optional[float] = None

MAX_ARTEFACT_DURATION: Optional[float] = None

OUTPUT_DIR: Optional[str] = None
INPUT_DIR: Optional[str] = None
DATABASE_DIR: Optional[str] = None

DISTANCE_SLINGER: Optional[float] = None
PROMINENCE_SLINGER: Optional[float] = None

CALIBRATION_STD_THRESHOLD: Optional[float] = None
CALIBRATION_SIGNAL_THRESHOLD: Optional[float] = None

FLUSH_STD_THRESHOLD: Optional[float] = None
FLUSH_SIGNAL_THRESHOLD: Optional[float] = None

BLOCK_STD_THRESHOLD: Optional[float] = None

ARTEFACT_PRIORITY: Optional[Dict[str, int]] = None

def init():
    """
    Initialize global settings for artefact detection.
    """

    global SAMPLING_DT
    global SAMPLING_FREQUENCY

    global MAX_ARTEFACT_DURATION

    global OUTPUT_DIR
    global INPUT_DIR
    global DATABASE_DIR

    global DISTANCE_SLINGER
    global PROMINENCE_SLINGER

    global CALIBRATION_STD_THRESHOLD
    global CALIBRATION_SIGNAL_THRESHOLD

    global FLUSH_STD_THRESHOLD
    global FLUSH_SIGNAL_THRESHOLD

    global BLOCK_STD_THRESHOLD

    global ARTEFACT_PRIORITY


    # ===== Measurement settings ===================================================================
    SAMPLING_DT = 0.008 # seconds
    SAMPLING_FREQUENCY = 1/SAMPLING_DT

    # ===== File paths =============================================================================
    # output and input directories
    OUTPUT_DIR = r'E:\OneDrive\School\Technical Medicine\TM Jaar 2\Stage 1 - OLVG IC\Artefact_Detect\output'
    INPUT_DIR = r'E:\OneDrive\School\Technical Medicine\TM Jaar 2\Stage 1 - OLVG IC\Artefact_Detect\data'

    # database file path
    DATABASE_DIR = r'E:\OneDrive\School\Technical Medicine\TM Jaar 2\Stage 1 - OLVG IC\Artefact_Detect\output\database_001.pkl'

     # ===== Artefact detection parameters ==========================================================
    MAX_ARTEFACT_DURATION = 100 # seconds

    # Slinger
    DISTANCE_SLINGER = 0.2 # seconds
    PROMINENCE_SLINGER = 0.004 # amplitude

    # Calibration
    CALIBRATION_STD_THRESHOLD = 5 # mmHg/s
    CALIBRATION_SIGNAL_THRESHOLD = 20 # mmHg

    # Flush
    FLUSH_STD_THRESHOLD = 5 # mmHg/s
    FLUSH_SIGNAL_THRESHOLD = 210 # mmHg

    # Block
    BLOCK_STD_THRESHOLD = 2 # mmHg/s

    # ===== Artefact priority (not appointed = infinite) =============================================
    ARTEFACT_PRIORITY = {
    'calibration_artefacts': 1,
    'flush_artefacts': 2,
    'block_artefacts': 3,
    'slinger_artefacts': 4
    }