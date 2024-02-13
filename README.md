# thesis-research-2.0
A new tidied version of my thesis research codes and resources.

This repo is structured as follows (still updating):
thesis-research-2.0/
├── data/
│   ├── raw/
│   │   ├── 6dfgs/
│   │   ├── sdss/
│   │   ├── lamost/
│   │   ├── 2mass/
│   │   └── r_e_jrl/
│   ├── preprocessed/
│   │   ├── sky_coord/
│   │   └── spectrophoto/
│   ├── processed/
│   │   ├── veldisp_calibrated/
│   │   ├── zmags_cut/
│   │   └── etg_selected/
│   └── foundation/
│       ├── fp_sample/
│       ├── pv_sample/
│       ├── smoothing/
│       └── bulk_flow/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_config.py
│   │   ├── cosmoFunc.py
│   │   ├── calc_kcor.py
│   │   └── gal_ext.py
│   ├── get_coordinates.py
│   ├── preprocessing.py
│   ├── processing.py
│   ├── select_etg.py
│   └── fit_fp.py
├── img/
├── log/
├── requirements.txt
├── notebook.ipynb
└── README.md