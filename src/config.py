"""
Configuration constants for scalar predictors.
"""
import numpy as np

# Column names for the 31 input parameters
COLNAMES = [
    'lH', 'lP', 'lE',                                           # Layer thicknesses (nm)
    'muHh', 'muPh', 'muPe', 'muEe',                              # Mobilities (log10 m²/V/s)
    'NvH', 'NcH', 'NvE', 'NcE', 'NvP', 'NcP',                    # DOS (log10 m⁻³)
    'chiHh', 'chiHe', 'chiPh', 'chiPe', 'chiEh', 'chiEe',        # Energy levels (eV)
    'Wlm', 'Whm',                                                # Work functions (eV)
    'epsH', 'epsP', 'epsE',                                      # Permittivities
    'Gavg',                                                       # Generation rate (log10 m⁻⁴s⁻¹)
    'Aug', 'Brad',                                               # Auger, Radiative (log10)
    'Taue', 'Tauh',                                              # Lifetimes (log10 s)
    'vII', 'vIII'                                                # Surface velocities (log10 m⁴/s)
]

# Parameters that are stored in log10 scale (must be exponentiated for physics)
LOG_SCALE_PARAMS = {
    'muHh', 'muPh', 'muPe', 'muEe',          # Mobilities
    'NvH', 'NcH', 'NvE', 'NcE', 'NvP', 'NcP', # DOS
    'Gavg',                                   # Generation
    'Aug', 'Brad',                            # Recombination coeffs
    'Taue', 'Tauh',                           # Lifetimes
    'vII', 'vIII'                             # Surface velocities
}

# Linear scale parameters
LINEAR_SCALE_PARAMS = {
    'lH', 'lP', 'lE',                         # Thicknesses
    'chiHh', 'chiHe', 'chiPh', 'chiPe', 'chiEh', 'chiEe',  # Energies
    'Wlm', 'Whm',                             # Work functions
    'epsH', 'epsP', 'epsE'                    # Permittivities
}

# Physical constants
KB_T = 0.0259  # kT at 300K in eV
Q_E = 1.602e-19  # Elementary charge (C)

# Voltage grid for J-V curves
def build_voltage_grid():
    return np.concatenate([
        np.arange(0, 0.4 + 1e-8, 0.1),
        np.arange(0.425, 1.4 + 1e-8, 0.025)
    ]).astype(np.float32)

V_GRID = build_voltage_grid()
N_VOLTAGE_POINTS = len(V_GRID)  # 45 points

# Target names
TARGETS = ['Voc', 'Jsc', 'Vmpp', 'Jmpp', 'FF', 'PCE', 'Pmpp']

# HPO settings
HPO_N_TRIALS_NN = 200
HPO_N_TRIALS_LGBM = 300
HPO_TIMEOUT_SECONDS = 7200  # 2 hours per model
HPO_N_JOBS = -1  # Use all available cores

# Training settings
BATCH_SIZE = 4096
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# File paths (defaults, can be overridden)
DEFAULT_PARAMS_FILE = 'LHS_parameters_m.txt'
DEFAULT_IV_FILE = 'IV_m.txt'
