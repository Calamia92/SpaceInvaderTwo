from __future__ import annotations

from .config import FIGURE_DIR, OUTPUT_DIR, REPORT_PATH
from .data import load_reports
from .phases import phase0, phase1, phase2
from .report import Report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_reports()
    report = Report("Le Bureau d'Analyse Terrestre - Rapport")
    for result in [phase0(df), phase1(df), phase2(df)]:
        report.add(result.heading, result.markdown)
        print(f"[ok] {result.heading}")
    report.write(REPORT_PATH)

    print(f"Transmission chargee : {len(df)} releves")
    print(f"Rapport ecrit : {REPORT_PATH}")


if __name__ == "__main__":
    main()
