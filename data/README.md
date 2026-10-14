# Data

`python scripts/download_data.py` puts the CWRU `.mat` files here, in `data/raw/`.

If a download fails, get the files from
https://engineering.case.edu/bearingdatacenter/download-data-file
and save each one as `data/raw/<number>.mat` (for example `data/raw/97.mat`).
The required numbers are listed in `config.yaml`.

The raw files are not committed to git (see `.gitignore`).
