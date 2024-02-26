# thesis-research-2.0
A new tidied version of my thesis research codes and resources.

This repo is structured as follows (still updating):
```bash
thesis-research-2.0/
├── data/
│   ├── raw/
│   │   ├── 6dfgs/
│   │   ├── sdss/
│   │   ├── lamost/
│   │   ├── 2mass/
│   │   └── r_e_jrl/
│   ├── external/
│   │   └── tempel_group_sdss8/
│   ├── preprocessed/
│   │   ├── sky_coord/
│   │   └── spectrophoto/
│   ├── processed/
│   │   ├── rsi_derived/
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
│   │   ├── calc_kcor.py
│   │   ├── constants.py
│   │   ├── CosmoFunc.py
│   │   ├── dustmaps_downloader.py
│   │   ├── helio_cmb.py
│   │   └── logging_config.py
│   ├── 1_get_coordinates.py
│   ├── 2_combine_spectrophoto.py
│   ├── 3_derive_rsi.py
│   └── 4_veldisp_calibration.py
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