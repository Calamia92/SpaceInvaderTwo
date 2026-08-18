from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "RAPPORT.md"

RAW_CSV = DATA_DIR / "ufo_reports.csv"
DATA_URLS = [
    "https://raw.githubusercontent.com/planetsig/ufo-reports/master/csv-data/ufo-complete-geocoded-time-normalized.csv",
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2019/2019-06-25/ufo_sightings.csv",
]
