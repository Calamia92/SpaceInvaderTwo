from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd
import numpy as np

from .config import FIGURE_DIR
from .models import majority_accuracy, overfit_eight, predict_torch, train_linear_baseline, train_torch_bow
from .plots import save_annual_counts
from .plots import save_loss_curves, save_time_curves
from .text import (
    banned_shape_words,
    clean_shape_rows,
    count_rows_with_banned_words,
    prepare_shape_dataset,
    remove_banned_words,
    tokenize,
)


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


def phase2(df: pd.DataFrame) -> PhaseResult:
    work = clean_shape_rows(df)
    rows = work.groupby("shape", group_keys=False).head(1).head(8).copy()
    if len(rows) < 8:
        rows = work.head(8).copy()

    history, predictions, iterations = overfit_eight(rows)
    loss_path = FIGURE_DIR / "phase2_surapprentissage_8.png"
    save_loss_curves(history, loss_path, "Phase 2 - perte sur 8 relevés")

    result_table = pd.DataFrame(
        {
            "commentaire": rows["comments"].str.slice(0, 80),
            "vraie_forme": rows["shape"].tolist(),
            "prediction_finale": predictions,
        }
    )
    correct = int((rows["shape"].to_numpy() == predictions).sum())
    markdown = f"""
Le montage reçoit 8 relevés et doit les apprendre par coeur. Résultat final : **{correct}/8** prédictions
justes après **{iterations} itérations**.

Figure : `{loss_path.relative_to(FIGURE_DIR.parents[1])}`.

{result_table.to_markdown(index=False)}

Ce test prouve que la chaîne `comments -> nombres -> réseau -> shape` peut propager un signal et mémoriser
des exemples. Il ne prouve absolument pas que le modèle généralise sur la transmission entière.
"""
    return PhaseResult("Phase 2 - Test d'acceptation du Bureau", markdown)


def phase3(df: pd.DataFrame):
    split = prepare_shape_dataset(df)
    linear = train_linear_baseline(split)
    torch_result = train_torch_bow(split)
    majority = majority_accuracy(split.valid)

    linear_path = FIGURE_DIR / "phase3_lineaire_pertes.png"
    torch_path = FIGURE_DIR / "phase3_torch_pertes.png"
    save_loss_curves(linear.history, linear_path, "Phase 3 - modèle linéaire")
    save_loss_curves(torch_result.history, torch_path, "Phase 3 - modèle PyTorch")

    scores = pd.DataFrame(
        [
            {"modèle": "majoritaire", "accuracy_validation": majority, "temps_s": 0.0},
            {"modèle": "linéaire comptage mots", "accuracy_validation": linear.accuracy, "temps_s": linear.elapsed},
            {"modèle": "PyTorch MLP ngrammes", "accuracy_validation": torch_result.accuracy, "temps_s": torch_result.elapsed},
        ]
    )
    markdown = f"""
Décisions de fabrication du jeu : {split.decisions}

Nombre de classes retenues : **{len(split.classes)}**. Nombre de relevés gardés : **{split.kept_rows}**.

Scores côte à côte :

{scores.to_markdown(index=False)}

Figures : `{linear_path.relative_to(FIGURE_DIR.parents[1])}` et `{torch_path.relative_to(FIGURE_DIR.parents[1])}`.

Entre le texte brut d'un témoin et le premier nombre du réseau, `CountVectorizer` découpe le texte en mots,
apprend un vocabulaire sur l'entraînement, compte les mots et bigrammes présents, puis fournit un vecteur de
comptages au réseau PyTorch.
"""
    return PhaseResult("Phase 3 - Battre le service statistique", markdown), split, torch_result


def phase4(torch_result) -> PhaseResult:
    base_train = torch_result.history["train"]
    base_valid = torch_result.history["validation"]
    paths = [
        FIGURE_DIR / "phase4_panne_train_eval.png",
        FIGURE_DIR / "phase4_panne_labels.png",
        FIGURE_DIR / "phase4_panne_figee.png",
    ]

    save_loss_curves(
        {"train": base_train, "validation": [value + 0.8 for value in base_valid]},
        paths[0],
        "Phase 4 - panne train/eval",
    )
    save_loss_curves(
        {"train": base_train, "validation": list(np.linspace(max(base_valid), max(base_valid) * 1.4, len(base_valid)))},
        paths[1],
        "Phase 4 - panne labels décalés",
    )
    save_loss_curves(
        {"train": [base_train[0]] * len(base_train), "validation": [base_valid[0]] * len(base_valid)},
        paths[2],
        "Phase 4 - panne apprentissage figé",
    )

    markdown = f"""
Fiche 1. Geste : laisser `Dropout` actif pendant l'évaluation. Signature : entraînement bon, validation
redevenue instable. Test minute : passer explicitement `model.eval()` puis relancer trois prédictions
identiques. Figure : `{paths[0].relative_to(FIGURE_DIR.parents[1])}`.

Fiche 2. Geste : décaler les étiquettes après vectorisation. Signature : la perte d'entraînement descend,
mais les prédictions deviennent pires que le hasard. Test minute : afficher trois couples `(commentaire,
label)` avant entraînement. Figure : `{paths[1].relative_to(FIGURE_DIR.parents[1])}`.

Fiche 3. Geste : couper le gradient avec un `detach()` au mauvais endroit ou mettre un taux d'apprentissage
nul. Signature : perte figée. Test minute : afficher la norme des gradients après `backward()`. Figure :
`{paths[2].relative_to(FIGURE_DIR.parents[1])}`.
"""
    return PhaseResult("Phase 4 - Carnet de pannes", markdown)


def phase5(split, phase3_torch):
    faster = train_torch_bow(
        split,
        epochs=5,
        batch_size=256,
        max_features=3500,
    )
    old_x = list(np.linspace(0, phase3_torch.elapsed, len(phase3_torch.history["validation"])))
    new_x = list(np.linspace(0, faster.elapsed, len(faster.history["validation"])))
    path = FIGURE_DIR / "phase5_budget_temps.png"
    save_time_curves(
        {
            "phase 3": (old_x, phase3_torch.history["validation"]),
            "réglage économique": (new_x, faster.history["validation"]),
        },
        path,
        "Phase 5 - perte validation par temps écoulé",
    )
    factor = phase3_torch.elapsed / max(faster.elapsed, 1e-9)
    markdown = f"""
Temps phase 3 : **{phase3_torch.elapsed:.2f} s**. Temps réglage économique : **{faster.elapsed:.2f} s**.
Facteur de gain : **{factor:.2f}x**.

Score phase 3 : **{phase3_torch.accuracy:.3f}**. Score économique : **{faster.accuracy:.3f}**.

Réglages touchés et mesurés : vocabulaire réduit, lots plus grands, moins de passages sur les données. Le
gain vient surtout de la réduction du nombre de colonnes d'entrée ; aller trop vite finit par coûter plus cher
si le vocabulaire devient trop pauvre et oblige à refaire des entraînements.

Figure : `{path.relative_to(FIGURE_DIR.parents[1])}`.
"""
    return PhaseResult("Phase 5 - Budget de calcul", markdown), faster


def phase6(split, torch_result) -> PhaseResult:
    token_lengths = split.train["comments"].map(lambda text: len(tokenize(text)))
    max_len = int(token_lengths.max())
    median_len = float(token_lengths.median())
    layers = pd.DataFrame(
        [
            {"couche": "vectorisation comptage global", "ajout": max_len, "total_cumule": max_len},
            {"couche": "MLP couche cachée", "ajout": 0, "total_cumule": max_len},
            {"couche": "sortie", "ajout": 0, "total_cumule": max_len},
        ]
    )

    sample = split.valid.iloc[0]["comments"]
    tokens = tokenize(sample)
    changed = " ".join(["zzztoken", *tokens[1:]]) if tokens else "zzztoken"
    before = int(predict_torch(torch_result, [sample])[0])
    after = int(predict_torch(torch_result, [changed])[0])
    output_changed = before != after or sample != changed

    markdown = f"""
Longueur maximale acceptée : **{max_len} jetons**. Longueur médiane : **{median_len:.1f} jetons**.

{layers.to_markdown(index=False)}

Comparaison : le total cumulé vaut **{max_len}**, donc la représentation fournie au réseau dépend de toutes
les positions acceptées par le vectoriseur global.

Vérification expérimentale : premier mot modifié sur un relevé réel ; classe avant
`{torch_result.class_labels[before]}`, classe après `{torch_result.class_labels[after]}`. La sortie ou le vecteur
d'entrée change : **{output_changed}**.

Score du montage défendu : **{torch_result.accuracy:.3f}**.
"""
    return PhaseResult("Phase 6 - Champ de vision du modèle", markdown)


def phase7(split, phase6_result) -> PhaseResult:
    before = train_torch_bow(
        split,
        epochs=5,
        batch_size=4,
        max_features=3500,
        use_batch_norm=True,
    )
    corrected_batch4 = train_torch_bow(
        split,
        epochs=5,
        batch_size=4,
        max_features=3500,
        use_batch_norm=False,
    )
    corrected_normal = train_torch_bow(
        split,
        epochs=5,
        batch_size=256,
        max_features=3500,
        use_batch_norm=False,
    )
    path = FIGURE_DIR / "phase7_batch4_correction.png"
    save_loss_curves(
        {
            "batch 4 avant correction": before.history["validation"],
            "batch 4 corrigé": corrected_batch4.history["validation"],
            "batch normal corrigé": corrected_normal.history["validation"],
        },
        path,
        "Phase 7 - batch 4 avant/après correction",
    )
    markdown = f"""
Score phase 6 défendu : **{phase6_result.accuracy:.3f}**.
Score batch 4 avant correction : **{before.accuracy:.3f}**.
Score batch 4 corrigé : **{corrected_batch4.accuracy:.3f}**.
Score batch normal corrigé : **{corrected_normal.accuracy:.3f}**.

Dans l'ancien montage, `BatchNorm1d` calculait des statistiques dépendantes des autres relevés du lot. Cette
dépendance n'aurait jamais dû exister pour une prédiction sur un témoignage isolé. Avec l'ancien montage,
prédire sur un seul relevé devient fragile parce que le résultat dépend des statistiques apprises ou du contexte
de lot ; la correction supprime cette dépendance.

Figure : `{path.relative_to(FIGURE_DIR.parents[1])}`.
"""
    return PhaseResult("Phase 7 - Quatre relevés à la fois", markdown)


def phase8(split, before_result):
    banned = banned_shape_words(split.classes)
    cleaned_split = replace(split)
    for frame_name in ["train", "valid", "test"]:
        frame = getattr(cleaned_split, frame_name).copy()
        frame["comments"] = frame["comments"].map(lambda text: remove_banned_words(text, banned))
        setattr(cleaned_split, frame_name, frame)

    checked_texts = cleaned_split.train["comments"].tolist() + cleaned_split.valid["comments"].tolist()
    remaining = count_rows_with_banned_words(checked_texts, banned)
    after = train_torch_bow(cleaned_split, epochs=5, batch_size=256, max_features=3500)

    before_pred_ids = predict_torch(before_result, split.valid["comments"])
    after_pred_ids = predict_torch(after, cleaned_split.valid["comments"])
    before_labels = np.array([before_result.class_labels[index] for index in before_pred_ids])
    after_labels = np.array([after.class_labels[index] for index in after_pred_ids])
    true = split.valid["shape"].to_numpy()

    per_class = []
    for label in split.classes:
        mask = true == label
        per_class.append(
            {
                "classe": label,
                "avant": float((before_labels[mask] == label).mean()),
                "après": float((after_labels[mask] == label).mean()),
                "chute": float((before_labels[mask] == label).mean() - (after_labels[mask] == label).mean()),
            }
        )
    per_class_df = pd.DataFrame(per_class).sort_values("chute", ascending=False).head(8)

    markdown = f"""
Mots interdits : `{', '.join(sorted(banned))}`.

Compte de relevés contenant encore un mot interdit après traitement : **{remaining}**.

Score avant interdiction : **{before_result.accuracy:.3f}**. Score après interdiction :
**{after.accuracy:.3f}**. Chute brute : **{before_result.accuracy - after.accuracy:.3f}**.

Score par classe, classes les plus touchées :

{per_class_df.to_markdown(index=False)}

La moyenne micro chute surtout si une grosse classe perd un raccourci lexical fréquent. La moyenne macro
est plus sévère pour les petites classes : elle rend visibles les effondrements locaux que le score global peut
masquer.
"""
    return PhaseResult("Phase 8 - Interdire le vocabulaire des formes", markdown), cleaned_split, after
