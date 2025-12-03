import numpy as np
import gnss_lib_py as glp

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.grace as grace

# %% ---------------------
# SECTION: Load GNSS Data
# ------------------------
# import data
# field 1 -- FEAR IT!!!!
field1 = "data/gnss_log_2025_11_18_11_29_26.txt"
# field 2 is whole sideline perimeter
field2 = "data/gnss_log_2025_11_18_11_21_46.txt"
# field 3 is only fraction of sideline distance
field3 = "data/gnss_log_2025_11_18_11_19_48.txt"
oval = "data/gnss_log_2025_11_18_11_49_34.txt"

# initial guess is Johnson Field Google Maps coords to help converge
x_init_guess_lla_J = np.array([[37.43109060346768], [-122.15524075735037], [0]])
x_init_guess_ecef_J = glp.geodetic_to_ecef(x_init_guess_lla_J).flatten()

# initial guess for Oval Google Maps coords to help converge
x_init_guess_lla_O = np.array([[37.43087105248864], [-122.16913436591264], [0]])
x_init_guess_ecef_O = glp.geodetic_to_ecef(x_init_guess_lla_O).flatten()

# NMEA data
nmea_field1 = "data/gnss_log_2025_11_18_11_29_26.nmea"
# field 2 is whole sideline perimeter
nmea_field2 = "data/gnss_log_2025_11_18_11_21_46.nmea"
# field 3 is only fraction of sideline distance
nmea_field3 = "data/gnss_log_2025_11_18_11_19_48.nmea"
nmea_oval = "data/gnss_log_2025_11_18_11_49_34.nmea"

# initial guess is Johnson Field Google Maps coords to help converge
x_init_guess_lla_J = np.array([[37.43109060346768], [-122.15524075735037], [0]])
x_init_guess_ecef_J = glp.geodetic_to_ecef(x_init_guess_lla_J).flatten()

# initial guess for Oval Google Maps coords to help converge
x_init_guess_lla_O = np.array([[37.43087105248864], [-122.16913436591264], [0]])
x_init_guess_ecef_O = glp.geodetic_to_ecef(x_init_guess_lla_O).flatten()
Results_NR_field1, Results_WLS_field1_cn0 = grace.run(exp=field1,
                                            x_init_guess_ecef=x_init_guess_ecef_J,
                                            max_iterations=10,
                                            tolerance=0.01,
                                            el_mask_cut=10,
                                            cn0_cut=20,
                                            w_scheme = 'cn0')

