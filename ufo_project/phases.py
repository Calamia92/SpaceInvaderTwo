from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import FIGURE_DIR
from .plots import save_annual_counts


@dataclass
class PhaseResult:
    heading: str
    markdown: str


def phase0(df: pd.DataFrame) -> PhaseResult:
    dated = df.dropna(subset=["observed_at"]).copy()
    dated = dated[(dated["observed_at"].dt.year >= 1990) & (dated["observed_at"].dt.year <= 2014)]
    days = dated["observed_at"].dt.normalize()
    day_counts = days.value_counts().sort_index()
    year_counts = dated.groupby(dated["observed_at"].dt.year).size()

    first_day = days.min().date()
    last_day = days.max().date()
    covered_days = (last_day - first_day).days + 1
    average_per_day = len(dated) / covered_days

    july_four = dated[(dated["observed_at"].dt.month == 7) & (dated["observed_at"].dt.day == 4)]
    july_four_per_year = july_four.groupby(july_four["observed_at"].dt.year).size()
    july_four_average = july_four_per_year.mean()

    weekday_pct = dated["observed_at"].dt.day_name().value_counts(normalize=True) * 100
    month_pct = dated["observed_at"].dt.month_name().value_counts(normalize=True) * 100

    max_day_count = int(day_counts.max())
    july_4_days = day_counts[(day_counts.index.month == 7) & (day_counts.index.day == 4)]
    best_july_4 = int(july_4_days.max())
    july_4_rank = int((day_counts.sort_values(ascending=False) > best_july_4).sum() + 1)

    figure_path = FIGURE_DIR / "phase0_volume_annuel.png"
    save_annual_counts(year_counts, figure_path)

    top10 = day_counts.sort_values(ascending=False).head(10).rename_axis("date").reset_index(name="releves")
    top10_md = top10.to_markdown(index=False)

    continuous_growth = all(year_counts.diff().dropna() > 0)
    markdown = f"""
Date utilisée : `datetime`, la date d'observation. C'est elle qui répond à la question du dossier : combien
de témoins regardaient le ciel un jour donné. `date_posted` mesure la publication administrative et ne répond
pas à cette question.

La transmission filtrée 1990-2014 couvre **{covered_days} jours**, du **{first_day}** au **{last_day}**.
Ce chiffre répond à la taille de la fenêtre temporelle étudiée.

Elle contient **{len(dated):,} relevés**, soit **{average_per_day:.1f} relevés par jour**. Cette moyenne répond
au niveau ordinaire de signalements par jour.

Un 4 juillet produit en moyenne **{july_four_average:.1f} relevés**. Ce chiffre répond à la charge typique
d'une date précise, pas à ce que les témoins ont réellement vu.

Le samedi porte **{weekday_pct.get("Saturday", 0):.1f} %** des relevés et le lundi **{weekday_pct.get("Monday", 0):.1f} %**.
Juillet porte **{month_pct.get("July", 0):.1f} %** des relevés et février **{month_pct.get("February", 0):.1f} %**.
Ces chiffres répondent aux biais de calendrier.

Le maximum atteint en une journée est **{max_day_count} relevés**. Le meilleur 4 juillet compte
**{best_july_4} relevés** et se classe au rang **{july_4_rank}** des journées les plus chargées.

Croissance annuelle strictement continue sur la série brute retenue : **{continuous_growth}**.

Figure : `{figure_path.relative_to(FIGURE_DIR.parents[1])}`.

Dix journées les plus chargées :

{top10_md}
"""
    return PhaseResult("Phase 0 - Refaire les calculs du disparu", markdown)


def _sample_report(df: pd.DataFrame, keyword: str, offset: int = 0) -> pd.Series:
    candidates = df[df["comments"].str.contains(keyword, case=False, regex=False, na=False)]
    candidates = candidates[candidates["comments"].str.len() > 40]
    if candidates.empty:
        candidates = df[df["comments"].str.len() > 40]
    return candidates.iloc[min(offset, len(candidates) - 1)]


def phase1(df: pd.DataFrame) -> PhaseResult:
    examples = [
        _sample_report(df, "firework", 0),
        _sample_report(df, "sound", 2),
        _sample_report(df, "triangle", 4),
    ]
    lines = []
    for row in examples:
        observed = row["observed_at"].date() if pd.notna(row["observed_at"]) else "date inconnue"
        shape = row["shape"] or "vide"
        lines.append(f"- `{observed}` / forme `{shape}` : {row['comments']}")

    markdown = f"""
Le chiffre du 4 juillet disait réellement qu'un volume inhabituel de relevés est associé à cette date. Il ne
disait pas que tous les témoins avaient vu la même chose, ni que la population ignorerait une flotte. Le même
chiffre autorise aussi une explication par le nombre de personnes dehors, par les feux d'artifice, ou par un
biais de déclaration sur une date facile à mémoriser.

Trois relevés recopiés depuis la transmission, choisis pour montrer ce qu'un comptage ne voit pas :

{chr(10).join(lines)}

Commande passée au système : **entrée** : le texte `comments` écrit par un témoin ; **sortie** : la forme
`shape` normalisée. La question que le système doit trancher est : *quelle forme observée est décrite par ce
témoignage ?* Un comptage de dates ne peut pas répondre à cette question, parce qu'il ne lit jamais les mots
des témoins.
"""
    return PhaseResult("Phase 1 - Le chiffre était vrai, la flotte est perdue", markdown)
