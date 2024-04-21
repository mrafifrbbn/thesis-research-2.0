# # Step 1: Combine spectroscopy+photometry data
# python src/2_combine_spectrophoto.py

# # Step 2: Derive r, s, i quantities
# python src/3_derive_rsi.py

# # Step 3: Calibrate velocity dispersions
# python src/4_veldisp_calibration.py

# Step 4: Apply selection criteria
python src/5_apply_selection.py

# Step 5: Calculate photometric errors and select FP samples
python src/6_calculate_phot_errors.py

# Step 6: Fit the Fundamental Plane
python src/7_fit_fp.py