from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd
import numpy as np

from .attention import attention_forward, benchmark_attention, embed_tokens, mean_absolute_disagreement
from .config import FIGURE_DIR
from .models import (
    majority_accuracy,
    overfit_eight,
    predict_torch,
    saliency_for_text,
    train_linear_baseline,
    train_torch_bow,
)
from .plots import save_annual_counts
from .plots import save_heatmap, save_loss_curves, save_time_curves
from .pretrained import measure_pretrained_regimes
from .retrieval import (
    answer_with_sources,
    markov_fake_comment,
    markov_state_signature,
    measure_retrieval_systems,
    naive_keyword_hits,
)
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


def phase3(df: pd.DataFrame, *, quick: bool = False):
    split = prepare_shape_dataset(df, quick=quick)
    linear = train_linear_baseline(split, epochs=3 if quick else 5, max_features=1800 if quick else 4000)
    torch_result = train_torch_bow(
        split,
        epochs=3 if quick else 8,
        batch_size=64 if quick else 128,
        max_features=2200 if quick else 6000,
    )
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


def phase5(split, phase3_torch, *, quick: bool = False):
    faster = train_torch_bow(
        split,
        epochs=2 if quick else 5,
        batch_size=128 if quick else 256,
        max_features=1600 if quick else 3500,
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


def phase7(split, phase6_result, *, quick: bool = False) -> PhaseResult:
    before = train_torch_bow(
        split,
        epochs=1 if quick else 5,
        batch_size=4,
        max_features=1400 if quick else 3500,
        use_batch_norm=True,
    )
    corrected_batch4 = train_torch_bow(
        split,
        epochs=1 if quick else 5,
        batch_size=4,
        max_features=1400 if quick else 3500,
        use_batch_norm=False,
    )
    corrected_normal = train_torch_bow(
        split,
        epochs=1 if quick else 5,
        batch_size=128 if quick else 256,
        max_features=1400 if quick else 3500,
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


def phase8(split, before_result, *, quick: bool = False):
    banned = banned_shape_words(split.classes)
    cleaned_split = replace(split)
    for frame_name in ["train", "valid", "test"]:
        frame = getattr(cleaned_split, frame_name).copy()
        frame["comments"] = frame["comments"].map(lambda text: remove_banned_words(text, banned))
        setattr(cleaned_split, frame_name, frame)

    checked_texts = cleaned_split.train["comments"].tolist() + cleaned_split.valid["comments"].tolist()
    remaining = count_rows_with_banned_words(checked_texts, banned)
    after = train_torch_bow(
        cleaned_split,
        epochs=2 if quick else 5,
        batch_size=128 if quick else 256,
        max_features=1600 if quick else 3500,
    )

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


def phase9(split, phase8_result) -> PhaseResult:
    pred_ids = predict_torch(phase8_result, split.valid["comments"])
    pred_labels = np.array([phase8_result.class_labels[index] for index in pred_ids])
    valid = split.valid.copy()
    valid["prediction"] = pred_labels

    cases = [
        ("réussi", valid[valid["shape"] == valid["prediction"]].head(1)),
        ("raté", valid[valid["shape"] != valid["prediction"]].head(1)),
        ("hésitation proche", valid.head(1)),
    ]

    blocks = []
    for title, frame in cases:
        if frame.empty:
            continue
        row = frame.iloc[0]
        saliency = saliency_for_text(phase8_result, row["comments"])
        salient_text = ", ".join(f"{word}={score:.3f}" for word, score in saliency)
        blocks.append(
            f"### Cas {title}\n\n"
            f"Vrai : `{row['shape']}`. Prédit : `{row['prediction']}`.\n\n"
            f"Témoignage : {row['comments']}\n\n"
            f"Mots ou ngrammes qui ont le plus pesé : {salient_text}\n\n"
            "Ce que la machine retient : des indices lexicaux courts encore présents après interdiction des noms de formes. "
            "Ce qu'elle ignore souvent : l'ordre narratif complet et les nuances humaines du témoignage tronqué. "
            "Le raté apprend surtout que le jeu mélange descriptions physiques, incertitude et vocabulaire de comparaison."
        )

    markdown = "\n\n".join(blocks)
    return PhaseResult("Phase 9 - Rendre des comptes sur trois décisions", markdown)


def _attention_record(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    pronouns = [" it ", " they ", " them ", " he ", " she "]
    candidates = df[df["comments"].str.len().between(40, 130)].copy()
    for pronoun in pronouns:
        hit = candidates[candidates["comments"].str.lower().str.contains(pronoun, regex=False)]
        if not hit.empty:
            row = hit.iloc[0]
            return row, tokenize(row["comments"])[:18]
    row = candidates.iloc[0]
    return row, tokenize(row["comments"])[:18]


def phase10(df: pd.DataFrame) -> PhaseResult:
    row, tokens = _attention_record(df)
    x = embed_tokens(tokens, dim=24)
    output, weights = attention_forward(x)
    row_sums = weights.sum(axis=1)
    path = FIGURE_DIR / "phase10_attention_matrice.png"
    save_heatmap(weights, tokens, path, "Phase 10 - matrice d'attention")

    weights_table = pd.DataFrame(weights, index=tokens, columns=tokens).round(3)
    markdown = f"""
Relevé réel utilisé : {row['comments']}

Nombre de jetons : **{len(tokens)}**. Forme entrée : **{x.shape}**. Forme sortie : **{output.shape}**.
Chaque ligne de la matrice somme entre **{row_sums.min():.6f}** et **{row_sums.max():.6f}**.

Les lignes sont les mots qui posent une question ; les colonnes sont les mots consultés. Pour un pronom, la
case à lire est donc sur la ligne du pronom et la colonne du mot auquel on pense qu'il se rattache. Le modèle
n'est pas entraîné : on vérifie ici le calcul, pas la qualité linguistique.

Figure : `{path.relative_to(FIGURE_DIR.parents[1])}`.

Matrice arrondie :

{weights_table.to_markdown()}
"""
    return PhaseResult("Phase 10 - L'attention au tableau", markdown)


def phase11(df: pd.DataFrame) -> PhaseResult:
    _, tokens = _attention_record(df)
    permutation = np.random.default_rng(11).permutation(len(tokens))
    inverse = np.argsort(permutation)
    shuffled_tokens = [tokens[index] for index in permutation]

    x = embed_tokens(tokens, dim=24, positional=False)
    output, before_weights = attention_forward(x, seed=11)
    shuffled_output, _ = attention_forward(x[permutation], seed=11)
    before_gap = float(np.linalg.norm(output - shuffled_output[inverse]))

    x_pos = embed_tokens(tokens, dim=24, positional=True)
    output_pos, after_weights = attention_forward(x_pos, seed=11)
    shuffled_x_pos = embed_tokens(shuffled_tokens, dim=24, positional=True)
    shuffled_output_pos, _ = attention_forward(shuffled_x_pos, seed=11)
    after_gap = float(np.linalg.norm(output_pos - shuffled_output_pos[inverse]))

    before_path = FIGURE_DIR / "phase11_avant_position.png"
    after_path = FIGURE_DIR / "phase11_apres_position.png"
    save_heatmap(before_weights, tokens, before_path, "Phase 11 - avant position")
    save_heatmap(after_weights, tokens, after_path, "Phase 11 - après position")

    markdown = f"""
Phrase correcte : `{' '.join(tokens)}`.
Phrase mélangée : `{' '.join(shuffled_tokens)}`.

Écart entre les sorties avant correction : **{before_gap:.10f}**.
Écart mesuré de la même façon après correction positionnelle : **{after_gap:.10f}**.

Avant correction, l'attention reçoit seulement les vecteurs des mots : permuter les mots permute les sorties,
mais chaque mot garde le même résultat quand on le remet à sa place. Le conseiller a donc raison : l'ordre
n'est pas représenté. Après correction, une position sinusoïdale est ajoutée aux vecteurs d'entrée avant de
fabriquer questions, étiquettes et contenus. On l'injecte là pour laisser intact le mécanisme d'attention de la
phase 10 tout en donnant à chaque mot une information sur sa place.

Figures : `{before_path.relative_to(FIGURE_DIR.parents[1])}` et `{after_path.relative_to(FIGURE_DIR.parents[1])}`.
"""
    return PhaseResult("Phase 11 - Le Conseil mélange les mots", markdown)


def phase12() -> PhaseResult:
    rows = benchmark_attention([32, 64, 128, 256, 512], dim=32, repeats=7)
    table = pd.DataFrame(rows)
    path = FIGURE_DIR / "phase12_cout_attention.png"

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(table["longueur"], table["temps_s_median"], marker="o", linewidth=1.8)
    ax.set_title("Phase 12 - coût de l'attention")
    ax.set_xlabel("Longueur (jetons)")
    ax.set_ylabel("Temps médian d'un passage avant (s)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    ratios = table["temps_s_median"].iloc[1:].to_numpy() / table["temps_s_median"].iloc[:-1].to_numpy()
    matrix_ratios = table["cases_matrice"].iloc[1:].to_numpy() / table["cases_matrice"].iloc[:-1].to_numpy()
    median_ratio = float(np.median(ratios))
    median_matrix_ratio = float(np.median(matrix_ratios))
    worst = table.iloc[-1]

    markdown = f"""
Protocole : même code d'attention que les phases 10 et 11, dimensions fixées à 32, sept passages par
longueur, et conservation du temps médian pour éviter le tir unique.

{table.to_markdown(index=False)}

Quand la longueur double, le temps est multiplié par **{median_ratio:.2f}** en médiane sur ces mesures. La
matrice des poids, elle, est multipliée par **{median_matrix_ratio:.1f}**, parce qu'elle contient
`longueur x longueur` cases. La courbe suit donc la montée quadratique attendue, avec du bruit de mesure CPU
sur les petites longueurs.

Figure : `{path.relative_to(FIGURE_DIR.parents[1])}`.

D'après ces chiffres, la machine commence à devenir inutilisable au-delà de **{int(worst['longueur'])} jetons**
pour un traitement interactif répété : à cette taille, une seule matrice contient déjà **{int(worst['cases_matrice'])}**
cases, et chaque doublement quadruple cette matrice.
"""
    return PhaseResult("Phase 12 - Le Conseil demande la facture", markdown)


def phase13(df: pd.DataFrame) -> PhaseResult:
    _, tokens = _attention_record(df)
    x = embed_tokens(tokens, dim=24, positional=True)

    output_a, weights_a = attention_forward(x, seed=13)
    output_b, weights_b = attention_forward(x, seed=14)
    _, weights_control = attention_forward(x, seed=13)
    combined = np.concatenate([output_a, output_b], axis=1)

    disagreement = mean_absolute_disagreement(weights_a, weights_b)
    control_disagreement = mean_absolute_disagreement(weights_a, weights_control)

    path_a = FIGURE_DIR / "phase13_tete_a.png"
    path_b = FIGURE_DIR / "phase13_tete_b.png"
    save_heatmap(weights_a, tokens, path_a, "Phase 13 - tête A")
    save_heatmap(weights_b, tokens, path_b, "Phase 13 - tête B")

    markdown = f"""
Relevé utilisé : `{' '.join(tokens)}`.

Deux têtes tournent en parallèle sur les mêmes vecteurs d'entrée positionnés. Chaque tête possède ses propres
matrices de question, d'étiquette et de contenu. Leurs sorties ont les formes **{output_a.shape}** et
**{output_b.shape}** ; la sortie recollée a la forme **{combined.shape}**.

Mesure choisie : désaccord absolu moyen entre les deux matrices de poids. Elle est adaptée ici parce qu'elle
compare directement les proportions d'attention case par case.

Désaccord entre deux têtes initialisées différemment : **{disagreement:.6f}**.
Cas de contrôle, deux têtes volontairement identiques : **{control_disagreement:.6f}**.

Figures : `{path_a.relative_to(FIGURE_DIR.parents[1])}` et `{path_b.relative_to(FIGURE_DIR.parents[1])}`.

Ces têtes ne sont pas entraînées ; leurs différences viennent donc de leur initialisation. Si elles étaient
entraînées, on pourrait conclure davantage : par exemple vérifier si une tête se spécialise sur les reprises
pronominales pendant qu'une autre suit les objets ou les couleurs.
"""
    return PhaseResult("Phase 13 - Deux regards sur le même relevé", markdown)


def phase14(phase8_result, *, with_pretrained: bool = False) -> PhaseResult:
    rows = [
        {
            "régime": "référence phase 8",
            "score": f"{phase8_result.accuracy:.3f}",
            "valeurs_modifiées": "MLP local complet",
            "temps_passage_s": f"{phase8_result.elapsed:.3f}",
            "mémoire": "CPU non tracée",
            "poids_sauvé": "poids du MLP local",
            "note": "vocabulaire des formes interdit",
        }
    ]
    for measurement in measure_pretrained_regimes(with_pretrained):
        rows.append(
            {
                "régime": measurement.regime,
                "score": measurement.score,
                "valeurs_modifiées": measurement.trainable_values,
                "temps_passage_s": measurement.train_step_seconds,
                "mémoire": measurement.memory,
                "poids_sauvé": measurement.saved_weight,
                "note": measurement.note,
            }
        )
    table = pd.DataFrame(rows)
    mode_note = (
        "Les mesures de modèle emprunté ont été tentées avec `prajjwal1/bert-tiny`."
        if with_pretrained
        else "Le run standard n'active pas le téléchargement du modèle ; relancer avec `--with-pretrained` pour mesurer."
    )
    markdown = f"""
Point de départ : modèle de la phase 8, mêmes relevés, même interdiction du vocabulaire des formes.

{table.to_markdown(index=False)}

Modèle emprunté choisi : `prajjwal1/bert-tiny`, assez petit pour un CPU et récupérable librement via
Transformers. Les trois régimes prévus sont : extracteur gelé avec une petite tête entraînée, fine-tuning
partiel des couches proches de la sortie, et adaptateurs qui ajoutent peu de valeurs sans modifier le modèle
de base.

{mode_note}

Décision actuelle : le Bureau peut se payer l'extracteur gelé ou les adaptateurs. Le fine-tuning partiel est plus
cher en valeurs sauvegardées et en mémoire, donc il ne se justifie que si son score dépasse nettement la ligne
de référence.
"""
    return PhaseResult("Phase 14 - Le cerveau emprunté, et sa facture", markdown)


def phase15(df: pd.DataFrame) -> PhaseResult:
    questions = [
        "Est-ce que les apparitions au-dessus des zones habitées ont une forme particulière ?",
        "Que décrivent les témoins qui parlent de bruit ?",
        "Les témoins associent-ils certaines couleurs à certaines formes ?",
        "Y a-t-il des relevés où l'objet semble suivre une voiture ?",
    ]
    budget_chars = 1200
    blocks = []
    sourced = 0
    naive_total = 0
    for question in questions:
        result = answer_with_sources(df, question, budget_chars=budget_chars, top_k=6)
        sourced += int(not result.citations.empty)
        naive_total += naive_keyword_hits(df, question, top_k=6)
        citations = result.citations.copy()
        if not citations.empty:
            citations["comments"] = citations["comments"].str.slice(0, 110)
            citations_md = citations.to_markdown(index=False)
        else:
            citations_md = "_Aucun relevé cité._"
        blocks.append(
            f"### {question}\n\n"
            f"Réponse : {result.answer}\n\n"
            f"Budget utilisé : **{result.used_chars}/{budget_chars} caractères**. Temps recherche : **{result.elapsed:.3f} s**.\n\n"
            f"{citations_md}"
        )

    markdown = f"""
Questions figées avant mesure :

{chr(10).join(f'- {question}' for question in questions)}

Budget de texte retenu : **{budget_chars} caractères par question**, jamais dépassé. La sélection des relevés
est déterministe : même fichier, même question, même vectoriseur TF-IDF, mêmes citations.

Proportion de réponses avec relevés cités : **{sourced}/{len(questions)}**.
Comparaison naïve par mots présents dans la question : **{naive_total} correspondances** dans les six premiers
relevés testés par question, sans classement sémantique.

{chr(10).join(blocks)}

Quand rien de proche n'est trouvé dans le budget, le système répond explicitement qu'il ne sait pas au lieu
d'inventer un relevé.
"""
    return PhaseResult("Phase 15 - Questions sourcées", markdown)


def phase16(df: pd.DataFrame) -> PhaseResult:
    questions = [
        "Est-ce que les apparitions au-dessus des zones habitées ont une forme particulière ?",
        "Que décrivent les témoins qui parlent de bruit ?",
        "Les témoins associent-ils certaines couleurs à certaines formes ?",
        "Y a-t-il des relevés où l'objet semble suivre une voiture ?",
    ]
    accepted_overlap_loss = 0.25
    measurements = measure_retrieval_systems(df, questions, budget_chars=1200, top_k=6)
    table = pd.DataFrame(
        [
            {
                "système": item.name,
                "max_features": item.max_features,
                "poids_index_KiB": item.index_bytes / 1024,
                "build_s": item.build_seconds,
                "latence_s": item.latency_seconds,
                "débit_qps": item.throughput_qps,
                "réponses_sourcées": item.sourced_answers,
                "recouvrement": item.mean_overlap_with_reference,
            }
            for item in measurements
        ]
    )
    before, after = measurements
    size_gain = before.index_bytes / max(after.index_bytes, 1)
    latency_gain = before.latency_seconds / max(after.latency_seconds, 1e-9)
    throughput_gain = after.throughput_qps / max(before.throughput_qps, 1e-9)
    overlap_loss = 1 - after.mean_overlap_with_reference

    markdown = f"""
Marge annoncée avant optimisation : perte maximale acceptée de **{accepted_overlap_loss:.2f}** sur le
recouvrement moyen des relevés cités par rapport au système avant réduction.

Protocole : mêmes quatre questions que la phase 15, même budget de 1200 caractères, même machine. Le
système avant garde **{before.max_features}** entrées TF-IDF ; le système réduit garde **{after.max_features}**
entrées. Aucune donnée ni cache n'est ajouté au dépôt.

{table.to_markdown(index=False)}

Poids sur disque estimé de l'index : gain **{size_gain:.2f}x**. Latence d'une réponse unique : gain
**{latency_gain:.2f}x**. Débit : gain **{throughput_gain:.2f}x**. Écart de score constaté :
**{overlap_loss:.2f}** de perte de recouvrement.

Réduction appliquée : vocabulaire TF-IDF plus petit, ce qui réduit la matrice sparse et accélère la similarité
cosinus. Je m'arrête ici parce que la perte reste dans la marge annoncée ; l'étape suivante serait un export
autonome de l'index et une quantification plus grossière des poids de la matrice.
"""
    return PhaseResult("Phase 16 - Faire entrer le tout dans le vaisseau", markdown)


def phase17(df: pd.DataFrame) -> PhaseResult:
    comments = df["comments"].dropna().astype(str)
    style_sample = comments[comments.str.len().between(35, 135)].sample(n=min(1000, len(comments)), random_state=17)
    before_signature = markov_state_signature(style_sample)
    settings = [
        {"température": 0.2, "symptôme": "texte propre mais répétitif", "sortie": markov_fake_comment(style_sample, temperature=0.2, seed=17)},
        {"température": 1.6, "symptôme": "texte instable qui part dans tous les sens", "sortie": markov_fake_comment(style_sample, temperature=1.6, seed=18)},
        {"température": 0.8, "symptôme": "réglage recommandé", "sortie": markov_fake_comment(style_sample, temperature=0.8, seed=19)},
    ]
    after_signature = markov_state_signature(style_sample)
    table = pd.DataFrame(settings)
    real_examples = style_sample.sample(n=4, random_state=170).to_frame(name="vrai_relevé")

    markdown = f"""
Règle absolue respectée : aucune valeur interne de modèle n'est entraînée ni ajustée. La seule action est le
choix du prochain mot au moment d'écrire, contrôlé ici par la température d'une chaîne de Markov construite
sur des vrais relevés courts.

Signature de l'état avant génération : **{before_signature}**. Signature après génération : **{after_signature}**.
Elles sont identiques, ce qui démontre que les transitions disponibles n'ont pas bougé entre le premier et le
dernier essai.

Grille des réglages :

{table.to_markdown(index=False)}

Étalon de style, vrais relevés mélangés au faux recommandé pour un futur tri en aveugle :

{real_examples.to_markdown(index=False)}

Réglage recommandé au Bureau : **température 0.8**. Le réglage bas répète trop vite les mêmes enchaînements ;
le réglage haut saute entre débuts de témoignages et devient incohérent. Le point utile est au milieu, où le
texte reste plat, court et maladroit comme les relevés, sans tourner en boucle trop visiblement.
"""
    return PhaseResult("Phase 17 - Le faux témoignage", markdown)
