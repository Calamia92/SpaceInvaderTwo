from __future__ import annotations

from .config import FIGURE_DIR, OUTPUT_DIR, REPORT_PATH
from .data import load_reports
from .phases import phase0, phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12
from .report import Report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_reports()
    report = Report("Le Bureau d'Analyse Terrestre - Rapport")
    phase3_result, split, torch_result = phase3(df)
    phase5_result, fast_result = phase5(split, torch_result)
    phase8_result, cleaned_split, banned_result = phase8(split, fast_result)
    for result in [
        phase0(df),
        phase1(df),
        phase2(df),
        phase3_result,
        phase4(torch_result),
        phase5_result,
        phase6(split, fast_result),
        phase7(split, fast_result),
        phase8_result,
        phase9(cleaned_split, banned_result),
        phase10(df),
        phase11(df),
        phase12(),
    ]:
        report.add(result.heading, result.markdown)
        print(f"[ok] {result.heading}")
    report.write(REPORT_PATH)

    print(f"Transmission chargee : {len(df)} releves")
    print(f"Rapport ecrit : {REPORT_PATH}")


if __name__ == "__main__":
    main()
