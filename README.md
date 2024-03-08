# thesis-research-2.0
A new tidied version of my thesis research codes and resources.

This repo is structured as follows (still updating):
```bash
thesis-research-2.0/
├── data/
│   ├── raw/                            # Raw data, no modification whatsoever
│   │   ├── 6dfgs/
│   │   ├── sdss/
│   │   ├── lamost/
│   │   ├── 2mass/
│   │   └── r_e_jrl/                    # John's GALFIT measurements
│   ├── external/                       # External data for complement and consistency checks
│   │   └── tempel_group_sdss8/
│   ├── preprocessed/                   # No data transformation, only aggregation and column selection
│   │   ├── sky_coord/
│   │   └── spectrophoto/
│   ├── processed/                      # Transformed data are stored here
│   │   ├── rsi_derived/
│   │   ├── veldisp_calibrated/
│   │   ├── zmags_cut/
│   │   └── etg_selected/
│   └── foundation/                     # Data for the main analysis
│       ├── fp_sample/
│       ├── pv_sample/
│       ├── smoothing/
│       └── bulk_flow/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── calc_kcor.py
│   │   ├── constants.py
│   │   ├── CosmoFunc.py
│   │   ├── dustmaps_downloader.py
│   │   ├── helio_cmb.py
│   │   └── logging_config.py
│   ├── 1_get_coordinates.py            # Fetch RA & DEC from each survey to be passed to 2MASS
│   ├── 2_combine_spectrophoto.py       # Combining spectroscopy and photometry measurements
│   ├── 3_derive_rsi.py                 # Derive FP variables
│   ├── 4_veldisp_calibration.py        # Calibrate velocity dispersions
│   └── 5_apply_selection.py            # Apply redshift, magnitude, and veldisp cut
├── docs/
│   ├── catatan.docx
│   └── Detailed Guide.docx
├── img/
├── log/
├── requirements.txt
├── .env.template
├── notebook.ipynb
└── README.md
```