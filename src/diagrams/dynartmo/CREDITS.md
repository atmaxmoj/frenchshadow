# Attribution

The midsagittal articulatory model and its anchor contour data (`data_sagi/`)
are derived from **DYNARTmo** by **Bernd J. Kröger**:

> Bernd J. Kröger, "DYNARTmo: A Dynamic Articulatory Model for Visualization of
> Speech Movement Patterns", arXiv:2507.20343.

Licensed under **Creative Commons Attribution 4.0 International (CC-BY 4.0)**
— https://creativecommons.org/licenses/by/4.0/

`model.py` is a numpy-only port of the static-visualization routines from the
supplementary notebook `articulatoryModel.ipynb`; `data_sagi/*.txt` are the
original anchor contours, unmodified. The French IPA → parameter table
(`phone_params.py`) and the SVG renderer (`__init__.py`) are our own work.

Per CC-BY, this attribution must remain and be surfaced wherever the diagrams
are shown (e.g. an app credits/about note).
