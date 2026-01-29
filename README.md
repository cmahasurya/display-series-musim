# Display Series Musim

Streamlit dashboard to visualize dasarian rainfall ensemble time series
from a multi-sheet Excel file.

## Features
- Supports Excel with sheets 1–27
- Preserves original DASARIAN order
- Ensemble time series plot
- Threshold line (default 50 mm)
- PMK summary table
- Export to Excel and CSV

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
