from __future__ import annotations

from .config import FIGURE_DIR, OUTPUT_DIR, REPORT_PATH
from .data import load_reports
from .report import Report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_reports()
    report = Report("Le Bureau d'Analyse Terrestre - Rapport")
    report.add(
        "Etat initial",
        f"Transmission chargee : {len(df)} releves. Les phases seront ajoutees une par une.",
    )
    report.write(REPORT_PATH)

    print(f"Transmission chargee : {len(df)} releves")
    print(f"Rapport ecrit : {REPORT_PATH}")


if __name__ == "__main__":
    main()
