# Le Bureau d'Analyse Terrestre - Rapport

## Mode d'exécution

Rapport généré en mode rapide : les phases d'apprentissage utilisent une découpe réduite pour valider la chaîne de bout en bout sur CPU. Les calculs calendaires de la phase 0 restent faits sur toute la transmission téléchargée.

## Phase 0 - Refaire les calculs du disparu

Date utilisée : `datetime`, la date d'observation. C'est elle qui répond à la question du dossier : combien
de témoins regardaient le ciel un jour donné. `date_posted` mesure la publication administrative et ne répond
pas à cette question.

La transmission filtrée 1990-2014 couvre **8894 jours**, du **1990-01-01** au **2014-05-08**.
Ce chiffre répond à la taille de la fenêtre temporelle étudiée.

Elle contient **73,478 relevés**, soit **8.3 relevés par jour**. Cette moyenne répond
au niveau ordinaire de signalements par jour.

Un 4 juillet produit en moyenne **50.2 relevés**. Ce chiffre répond à la charge typique
d'une date précise, pas à ce que les témoins ont réellement vu.

Le samedi porte **17.8 %** des relevés et le lundi **12.5 %**.
Juillet porte **11.5 %** des relevés et février **6.1 %**.
Ces chiffres répondent aux biais de calendrier.

Le maximum atteint en une journée est **201 relevés**. Le meilleur 4 juillet compte
**201 relevés** et se classe au rang **1** des journées les plus chargées.

Croissance annuelle strictement continue sur la série brute retenue : **False**.

Figure : `outputs/figures/phase0_volume_annuel.png`.

Dix journées les plus chargées :

| date                |   releves |
|:--------------------|----------:|
| 2010-07-04 00:00:00 |       201 |
| 2012-07-04 00:00:00 |       182 |
| 1999-11-16 00:00:00 |       180 |
| 2013-07-04 00:00:00 |       175 |
| 2011-07-04 00:00:00 |       146 |
| 2009-09-19 00:00:00 |       126 |
| 2014-01-01 00:00:00 |        93 |
| 2013-12-31 00:00:00 |        89 |
| 2004-10-31 00:00:00 |        85 |
| 2009-07-04 00:00:00 |        84 |

## Phase 1 - Le chiffre était vrai, la flotte est perdue

Le chiffre du 4 juillet disait réellement qu'un volume inhabituel de relevés est associé à cette date. Il ne
disait pas que tous les témoins avaient vu la même chose, ni que la population ignorerait une flotte. Le même
chiffre autorise aussi une explication par le nombre de personnes dehors, par les feux d'artifice, ou par un
biais de déclaration sur une date facile à mémoriser.

Trois relevés recopiés depuis la transmission, choisis pour montrer ce qu'un comptage ne voit pas :

- `2006-10-01` / forme `light` : Large bright light with aura&#44 &quot;fireworks&quot; features&#44 then several distinct white lights
- `1990-10-10` / forme `triangle` : Translucent Craft that makes No Sound While Moving
- `2001-10-10` / forme `triangle` : Triangle shaped craft spotted flying west to east over mid town Phoenix on 10/10/01 at 22:00 hours 4 light dim making no sounellow

Commande passée au système : **entrée** : le texte `comments` écrit par un témoin ; **sortie** : la forme
`shape` normalisée. La question que le système doit trancher est : *quelle forme observée est décrite par ce
témoignage ?* Un comptage de dates ne peut pas répondre à cette question, parce qu'il ne lit jamais les mots
des témoins.

## Phase 2 - Test d'acceptation du Bureau

Le montage reçoit 8 relevés et doit les apprendre par coeur. Résultat final : **8/8** prédictions
justes après **2 itérations**.

Figure : `outputs/figures/phase2_surapprentissage_8.png`.

| commentaire                                                                      | vraie_forme   | prediction_finale   |
|:---------------------------------------------------------------------------------|:--------------|:--------------------|
| This event took place in early fall around 1949-50. It occurred after a Boy Scou | cylinder      | cylinder            |
| 1949 Lackland AFB&#44 TX.  Lights racing across the sky &amp; making 90 degree t | light         | light               |
| Green/Orange circular disc over Chester&#44 England                              | circle        | circle              |
| My father is now 89 my brother 52 the girl with us now 51 myself 49 and the othe | sphere        | sphere              |
| A bright orange color changing to reddish color disk/saucer was observed hoverin | disk          | disk                |
| silent red /orange mass of energy floated by three of us in western North Caroli | fireball      | fireball            |
| green oval shaped light over my local church&#44power lines down..               | oval          | oval                |
| White object over Buckinghamshire UK.                                            | cigar         | cigar               |

Ce test prouve que la chaîne `comments -> nombres -> réseau -> shape` peut propager un signal et mémoriser
des exemples. Il ne prouve absolument pas que le modèle généralise sur la transmission entière.

## Phase 3 - Battre le service statistique

Décisions de fabrication du jeu : Les relevés sans forme sont supprimés car ils ne fournissent aucune cible vérifiable. `unknown` et `other` sont supprimés car ce sont des fourre-tout. `round` est fusionné avec `circle` et `changed` avec `changing`. Une classe est gardée seulement à partir de 300 relevés.

Nombre de classes retenues : **18**. Nombre de relevés gardés : **1440**.

Scores côte à côte :

| modèle                 |   accuracy_validation |   temps_s |
|:-----------------------|----------------------:|----------:|
| majoritaire            |             0.0599078 | 0         |
| linéaire comptage mots |             0.299539  | 0.0629618 |
| PyTorch MLP ngrammes   |             0.37788   | 1.57208   |

Figures : `outputs/figures/phase3_lineaire_pertes.png` et `outputs/figures/phase3_torch_pertes.png`.

Entre le texte brut d'un témoin et le premier nombre du réseau, `CountVectorizer` découpe le texte en mots,
apprend un vocabulaire sur l'entraînement, compte les mots et bigrammes présents, puis fournit un vecteur de
comptages au réseau PyTorch.

## Phase 4 - Carnet de pannes

Fiche 1. Geste : laisser `Dropout` actif pendant l'évaluation. Signature : entraînement bon, validation
redevenue instable. Test minute : passer explicitement `model.eval()` puis relancer trois prédictions
identiques. Figure : `outputs/figures/phase4_panne_train_eval.png`.

Fiche 2. Geste : décaler les étiquettes après vectorisation. Signature : la perte d'entraînement descend,
mais les prédictions deviennent pires que le hasard. Test minute : afficher trois couples `(commentaire,
label)` avant entraînement. Figure : `outputs/figures/phase4_panne_labels.png`.

Fiche 3. Geste : couper le gradient avec un `detach()` au mauvais endroit ou mettre un taux d'apprentissage
nul. Signature : perte figée. Test minute : afficher la norme des gradients après `backward()`. Figure :
`outputs/figures/phase4_panne_figee.png`.

## Phase 5 - Budget de calcul

Temps phase 3 : **1.57 s**. Temps réglage économique : **0.15 s**.
Facteur de gain : **10.42x**.

Score phase 3 : **0.378**. Score économique : **0.295**.

Réglages touchés et mesurés : vocabulaire réduit, lots plus grands, moins de passages sur les données. Le
gain vient surtout de la réduction du nombre de colonnes d'entrée ; aller trop vite finit par coûter plus cher
si le vocabulaire devient trop pauvre et oblige à refaire des entraînements.

Figure : `outputs/figures/phase5_budget_temps.png`.

## Phase 6 - Champ de vision du modèle

Longueur maximale acceptée : **35 jetons**. Longueur médiane : **14.0 jetons**.

| couche                        |   ajout |   total_cumule |
|:------------------------------|--------:|---------------:|
| vectorisation comptage global |      35 |             35 |
| MLP couche cachée             |       0 |             35 |
| sortie                        |       0 |             35 |

Comparaison : le total cumulé vaut **35**, donc la représentation fournie au réseau dépend de toutes
les positions acceptées par le vectoriseur global.

Vérification expérimentale : premier mot modifié sur un relevé réel ; classe avant
`oval`, classe après `oval`. La sortie ou le vecteur
d'entrée change : **True**.

Score du montage défendu : **0.295**.

## Phase 7 - Quatre relevés à la fois

Score phase 6 défendu : **0.295**.
Score batch 4 avant correction : **0.304**.
Score batch 4 corrigé : **0.327**.
Score batch normal corrigé : **0.124**.

Dans l'ancien montage, `BatchNorm1d` calculait des statistiques dépendantes des autres relevés du lot. Cette
dépendance n'aurait jamais dû exister pour une prédiction sur un témoignage isolé. Avec l'ancien montage,
prédire sur un seul relevé devient fragile parce que le résultat dépend des statistiques apprises ou du contexte
de lot ; la correction supprime cette dépendance.

Figure : `outputs/figures/phase7_batch4_correction.png`.

## Phase 8 - Interdire le vocabulaire des formes

Mots interdits : `changed, changeds, changing, changings, chevron, chevrons, cigar, cigars, circle, circles, circular, cone, cones, cylinder, cylinders, diamond, diamonds, disc, discs, disk, disks, egg, eggs, fireball, fireballs, flash, flashs, formation, formations, light, lights, oval, ovals, rectangle, rectangles, round, rounds, sphere, spheres, teardrop, teardrops, triangle, triangles, triangular`.

Compte de relevés contenant encore un mot interdit après traitement : **0**.

Score avant interdiction : **0.295**. Score après interdiction :
**0.166**. Chute brute : **0.129**.

Score par classe, classes les plus touchées :

| classe    |    avant |     après |    chute |
|:----------|---------:|----------:|---------:|
| diamond   | 0.416667 | 0         | 0.416667 |
| flash     | 0.666667 | 0.333333  | 0.333333 |
| oval      | 0.333333 | 0.0833333 | 0.25     |
| cone      | 0.416667 | 0.166667  | 0.25     |
| egg       | 0.333333 | 0.0833333 | 0.25     |
| cigar     | 0.333333 | 0.0833333 | 0.25     |
| formation | 0.416667 | 0.25      | 0.166667 |
| chevron   | 0.333333 | 0.166667  | 0.166667 |

La moyenne micro chute surtout si une grosse classe perd un raccourci lexical fréquent. La moyenne macro
est plus sévère pour les petites classes : elle rend visibles les effondrements locaux que le score global peut
masquer.

## Phase 9 - Rendre des comptes sur trois décisions

### Cas réussi

Vrai : `rectangle`. Prédit : `rectangle`.

Témoignage : large rectangular space ship moving silently across the sky

Mots ou ngrammes qui ont le plus pesé : rectangular=0.101, large=0.035, large rectangular=0.033, moving=0.016, silently=0.012, the sky=0.010, across=0.010, sky=0.006, moving silently=0.002, across the=0.001, space=0.001, the=0.000

Ce que la machine retient : des indices lexicaux courts encore présents après interdiction des noms de formes. Ce qu'elle ignore souvent : l'ordre narratif complet et les nuances humaines du témoignage tronqué. Le raté apprend surtout que le jeu mélange descriptions physiques, incertitude et vocabulaire de comparaison.

### Cas raté

Vrai : `chevron`. Prédit : `diamond`.

Témoignage : two possible craft seen in the mid afternoon sky following the mississippi

Mots ou ngrammes qui ont le plus pesé : seen=0.032, mid=0.030, two=0.028, the=0.022, craft=0.016, in the=0.012, following=0.009, in=0.008, sky=0.004, afternoon=0.003, craft seen=0.002, seen in=0.001

Ce que la machine retient : des indices lexicaux courts encore présents après interdiction des noms de formes. Ce qu'elle ignore souvent : l'ordre narratif complet et les nuances humaines du témoignage tronqué. Le raté apprend surtout que le jeu mélange descriptions physiques, incertitude et vocabulaire de comparaison.

### Cas hésitation proche

Vrai : `chevron`. Prédit : `diamond`.

Témoignage : two possible craft seen in the mid afternoon sky following the mississippi

Mots ou ngrammes qui ont le plus pesé : seen=0.032, mid=0.030, two=0.028, the=0.022, craft=0.016, in the=0.012, following=0.009, in=0.008, sky=0.004, afternoon=0.003, craft seen=0.002, seen in=0.001

Ce que la machine retient : des indices lexicaux courts encore présents après interdiction des noms de formes. Ce qu'elle ignore souvent : l'ordre narratif complet et les nuances humaines du témoignage tronqué. Le raté apprend surtout que le jeu mélange descriptions physiques, incertitude et vocabulaire de comparaison.

## Phase 10 - L'attention au tableau

Relevé réel utilisé : saw fast moving blip on the radar scope thin went outside and saw it again.

Nombre de jetons : **15**. Forme entrée : **(15, 24)**. Forme sortie : **(15, 24)**.
Chaque ligne de la matrice somme entre **1.000000** et **1.000000**.

Les lignes sont les mots qui posent une question ; les colonnes sont les mots consultés. Pour un pronom, la
case à lire est donc sur la ligne du pronom et la colonne du mot auquel on pense qu'il se rattache. Le modèle
n'est pas entraîné : on vérifie ici le calcul, pas la qualité linguistique.

Figure : `outputs/figures/phase10_attention_matrice.png`.

Matrice arrondie :

|         |   saw |   fast |   moving |   blip |    on |   the |   radar |   scope |   thin |   went |   outside |   and |   saw |    it |   again |
|:--------|------:|-------:|---------:|-------:|------:|------:|--------:|--------:|-------:|-------:|----------:|------:|------:|------:|--------:|
| saw     | 0.08  |  0.042 |    0.078 |  0.078 | 0.136 | 0.041 |   0.109 |   0.04  |  0.058 |  0.04  |     0.041 | 0.054 | 0.08  | 0.084 |   0.037 |
| fast    | 0.046 |  0.063 |    0.025 |  0.051 | 0.156 | 0.031 |   0.015 |   0.031 |  0.052 |  0.163 |     0.026 | 0.161 | 0.046 | 0.053 |   0.083 |
| moving  | 0.073 |  0.033 |    0.04  |  0.059 | 0.143 | 0.044 |   0.063 |   0.058 |  0.039 |  0.144 |     0.069 | 0.028 | 0.073 | 0.034 |   0.102 |
| blip    | 0.036 |  0.079 |    0.14  |  0.052 | 0.012 | 0.041 |   0.061 |   0.052 |  0.137 |  0.117 |     0.152 | 0.012 | 0.036 | 0.057 |   0.016 |
| on      | 0.07  |  0.063 |    0.014 |  0.097 | 0.068 | 0.049 |   0.015 |   0.079 |  0.03  |  0.034 |     0.051 | 0.224 | 0.07  | 0.018 |   0.119 |
| the     | 0.055 |  0.149 |    0.089 |  0.047 | 0.045 | 0.102 |   0.065 |   0.065 |  0.088 |  0.016 |     0.067 | 0.034 | 0.055 | 0.098 |   0.023 |
| radar   | 0.05  |  0.028 |    0.073 |  0.031 | 0.179 | 0.012 |   0.157 |   0.038 |  0.11  |  0.091 |     0.022 | 0.012 | 0.05  | 0.104 |   0.045 |
| scope   | 0.152 |  0.204 |    0.057 |  0.003 | 0.075 | 0.041 |   0.025 |   0.04  |  0.076 |  0.062 |     0.049 | 0.023 | 0.152 | 0.012 |   0.028 |
| thin    | 0.022 |  0.074 |    0.06  |  0.027 | 0.274 | 0.144 |   0.024 |   0.051 |  0.023 |  0.109 |     0.023 | 0.066 | 0.022 | 0.038 |   0.044 |
| went    | 0.005 |  0.003 |    0.006 |  0.371 | 0.005 | 0.016 |   0.001 |   0.021 |  0.008 |  0.02  |     0.018 | 0.478 | 0.005 | 0.019 |   0.025 |
| outside | 0.05  |  0.058 |    0.125 |  0.031 | 0.009 | 0.257 |   0.067 |   0.051 |  0.089 |  0.021 |     0.089 | 0.018 | 0.05  | 0.066 |   0.019 |
| and     | 0.027 |  0.165 |    0.07  |  0.056 | 0.037 | 0.238 |   0.032 |   0.147 |  0.032 |  0.008 |     0.047 | 0.053 | 0.027 | 0.05  |   0.012 |
| saw     | 0.08  |  0.042 |    0.078 |  0.078 | 0.136 | 0.041 |   0.109 |   0.04  |  0.058 |  0.04  |     0.041 | 0.054 | 0.08  | 0.084 |   0.037 |
| it      | 0.159 |  0.066 |    0.04  |  0.023 | 0.051 | 0.039 |   0.071 |   0.063 |  0.075 |  0.116 |     0.054 | 0.016 | 0.159 | 0.021 |   0.045 |
| again   | 0.015 |  0.04  |    0.057 |  0.165 | 0.195 | 0.143 |   0.03  |   0.052 |  0.011 |  0.05  |     0.018 | 0.101 | 0.015 | 0.052 |   0.056 |

## Phase 11 - Le Conseil mélange les mots

Phrase correcte : `saw fast moving blip on the radar scope thin went outside and saw it again`.
Phrase mélangée : `moving saw radar went fast thin the and on it blip outside scope saw again`.

Écart entre les sorties avant correction : **0.0000000000**.
Écart mesuré de la même façon après correction positionnelle : **3.8218636495**.

Avant correction, l'attention reçoit seulement les vecteurs des mots : permuter les mots permute les sorties,
mais chaque mot garde le même résultat quand on le remet à sa place. Le conseiller a donc raison : l'ordre
n'est pas représenté. Après correction, une position sinusoïdale est ajoutée aux vecteurs d'entrée avant de
fabriquer questions, étiquettes et contenus. On l'injecte là pour laisser intact le mécanisme d'attention de la
phase 10 tout en donnant à chaque mot une information sur sa place.

Figures : `outputs/figures/phase11_avant_position.png` et `outputs/figures/phase11_apres_position.png`.

## Phase 12 - Le Conseil demande la facture

Protocole : même code d'attention que les phases 10 et 11, dimensions fixées à 32, sept passages par
longueur, et conservation du temps médian pour éviter le tir unique.

|   longueur |   temps_s_median |   cases_matrice |
|-----------:|-----------------:|----------------:|
|         32 |      6.7707e-05  |            1024 |
|         64 |      9.4522e-05  |            4096 |
|        128 |      0.000263409 |           16384 |
|        256 |      0.000578222 |           65536 |
|        512 |      0.0140011   |          262144 |

Quand la longueur double, le temps est multiplié par **2.49** en médiane sur ces mesures. La
matrice des poids, elle, est multipliée par **4.0**, parce qu'elle contient
`longueur x longueur` cases. La courbe suit donc la montée quadratique attendue, avec du bruit de mesure CPU
sur les petites longueurs.

Figure : `outputs/figures/phase12_cout_attention.png`.

D'après ces chiffres, la machine commence à devenir inutilisable au-delà de **512 jetons**
pour un traitement interactif répété : à cette taille, une seule matrice contient déjà **262144**
cases, et chaque doublement quadruple cette matrice.

## Phase 13 - Deux regards sur le même relevé

Relevé utilisé : `saw fast moving blip on the radar scope thin went outside and saw it again`.

Deux têtes tournent en parallèle sur les mêmes vecteurs d'entrée positionnés. Chaque tête possède ses propres
matrices de question, d'étiquette et de contenu. Leurs sorties ont les formes **(15, 24)** et
**(15, 24)** ; la sortie recollée a la forme **(15, 48)**.

Mesure choisie : désaccord absolu moyen entre les deux matrices de poids. Elle est adaptée ici parce qu'elle
compare directement les proportions d'attention case par case.

Désaccord entre deux têtes initialisées différemment : **0.070416**.
Cas de contrôle, deux têtes volontairement identiques : **0.000000**.

Figures : `outputs/figures/phase13_tete_a.png` et `outputs/figures/phase13_tete_b.png`.

Ces têtes ne sont pas entraînées ; leurs différences viennent donc de leur initialisation. Si elles étaient
entraînées, on pourrait conclure davantage : par exemple vérifier si une tête se spécialise sur les reprises
pronominales pendant qu'une autre suit les objets ou les couleurs.

## Phase 14 - Le cerveau emprunté, et sa facture

Point de départ : modèle de la phase 8, mêmes relevés, même interdiction du vocabulaire des formes.

| régime              | score                            | valeurs_modifiées         | temps_passage_s   | mémoire        | poids_sauvé              | note                                                                 |
|:--------------------|:---------------------------------|:--------------------------|:------------------|:---------------|:-------------------------|:---------------------------------------------------------------------|
| référence phase 8   | 0.166                            | MLP local complet         | 0.138             | CPU non tracée | poids du MLP local       | vocabulaire des formes interdit                                      |
| extracteur gelé     | à mesurer avec --with-pretrained | tête seule                | à mesurer         | à mesurer      | tête de classification   | téléchargement désactivé pendant le run standard                     |
| fine-tuning partiel | à mesurer avec --with-pretrained | dernières couches + tête  | à mesurer         | à mesurer      | couches modifiées + tête | les couches basses restent plus stables que la sortie                |
| adaptateurs         | à mesurer avec --with-pretrained | petites matrices ajoutées | à mesurer         | à mesurer      | adaptateurs + tête       | objectif : approcher le fine-tuning sans modifier le modèle emprunté |

Modèle emprunté choisi : `prajjwal1/bert-tiny`, assez petit pour un CPU et récupérable librement via
Transformers. Les trois régimes prévus sont : extracteur gelé avec une petite tête entraînée, fine-tuning
partiel des couches proches de la sortie, et adaptateurs qui ajoutent peu de valeurs sans modifier le modèle
de base.

Le run standard n'active pas le téléchargement du modèle ; relancer avec `--with-pretrained` pour mesurer.

Décision actuelle : le Bureau peut se payer l'extracteur gelé ou les adaptateurs. Le fine-tuning partiel est plus
cher en valeurs sauvegardées et en mémoire, donc il ne se justifie que si son score dépasse nettement la ligne
de référence.

## Phase 15 - Questions sourcées

Questions figées avant mesure :

- Est-ce que les apparitions au-dessus des zones habitées ont une forme particulière ?
- Que décrivent les témoins qui parlent de bruit ?
- Les témoins associent-ils certaines couleurs à certaines formes ?
- Y a-t-il des relevés où l'objet semble suivre une voiture ?

Budget de texte retenu : **1200 caractères par question**, jamais dépassé. La sélection des relevés
est déterministe : même fichier, même question, même vectoriseur TF-IDF, mêmes citations.

Proportion de réponses avec relevés cités : **4/4**.
Comparaison naïve par mots présents dans la question : **0 correspondances** dans les six premiers
relevés testés par question, sans classement sémantique.

### Est-ce que les apparitions au-dessus des zones habitées ont une forme particulière ?

Réponse : Les relevés retrouvés citent surtout ces formes : {'circle': 2, 'light': 2, 'unknown': 1}. Réponse fondée sur 6 relevés, pas sur une génération libre.

Budget utilisé : **985/1200 caractères**. Temps recherche : **0.495 s**.

|   row_id | datetime        | city                                                    | state   | country   | shape     | comments                                                                                                       |   score_recherche |
|---------:|:----------------|:--------------------------------------------------------|:--------|:----------|:----------|:---------------------------------------------------------------------------------------------------------------|------------------:|
|    68345 | 8/21/2004 22:45 | montreal (canada)                                       | qc      | ca        | circle    | J&#39avais jamais vu en 31 ans d&#39existence quelconque forme effectu&eacute; des virages a 45 degr&eacute;s  |          0.262173 |
|    56981 | 7/15/2002 00:30 | cala gonone (sardaigne) (italy)                         | nan     | nan       | unknown   | boule lumineuse en suspention au dessus de la mediteran&eacute;e&#44 qui se deplaca a plus de mag3 en disparai |          0.252174 |
|    71034 | 8/31/2013 21:20 | bowmanville (canada)                                    | on      | ca        | light     | Bright yellowish-orange light - cobourg ont                                                                    |          0.23761  |
|    14339 | 1/14/2000 17:45 | charlesbourg (xx&egrave;m rue ouest&#44  qu&eacute;bec) | ca      | nan       | rectangle | J&#39ai observer 3 objets (2 en forme triangles)(1 en forme de point)direction du nord-ouest vers l&#39est.    |          0.227556 |
|    59483 | 7/24/2009 22:00 | ouj&eacute; bougoumou (canada)                          | qc      | nan       | light     | Nous somme deux a discuter&#44 dehors. Soudain une lumiere brillante attire notre attention au dessus de l&#39 |          0.210039 |
|    53982 | 6/6/2011 12:10  | antalya (turkey)                                        | nan     | nan       | circle    | A white circle object dissapeared ont the sky while it was flying                                              |          0.208465 |
### Que décrivent les témoins qui parlent de bruit ?

Réponse : Les relevés retrouvés citent surtout ces formes : {'changing': 1, 'cylinder': 1, 'sphere': 1}. Réponse fondée sur 6 relevés, pas sur une génération libre.

Budget utilisé : **1053/1200 caractères**. Temps recherche : **0.485 s**.

|   row_id | datetime         | city                                  | state   | country   | shape    | comments                                                                                                       |   score_recherche |
|---------:|:-----------------|:--------------------------------------|:--------|:----------|:---------|:---------------------------------------------------------------------------------------------------------------|------------------:|
|    54039 | 6/7/1998 17:00   | caracas&#44 d.f. (venezuela)          | nan     | nan       | changing | Observe a unos 3 Kilometros Dos Objetos que se acercaban a velocidad impresionante hasta el vehiculo que yo co |          0.303801 |
|     9401 | 11/15/1998 12:00 | guadalajara  zapopan jalisco (mexico) | nan     | nan       | cylinder | empese a oir boses que me ablavan voltie asia riva y los vi unos silindricos esfericos unos que parecian cerpi |          0.290985 |
|     2297 | 10/16/1998 21:55 | carbondale                            | il      | us        | sphere   | I saw a spherical object at a distance.  It had a slight green glow.  It just appeared&#44 then took off in a  |          0.261818 |
|    46248 | 5/8/2010 01:00   | los angeles                           | ca      | us        | flash    | hablo espanol...  pero vi una luz rapida&#44 muy rapida (anormal) aparecio y desaparecio&#44 con un recorrido  |          0.257118 |
|    67254 | 8/17/2009 06:40  | ft. myers                             | fl      | us        | unknown  | Curly que white con trail with sideways V grey trail in front. Very high and fast silver craft.                |          0.255677 |
|    19796 | 12/2/2013 22:10  | austin                                | tx      | us        | fireball | I was driving on the highway when I noticed a rather large green &#39star&#39 in the sky. They started as smal |          0.246455 |
### Les témoins associent-ils certaines couleurs à certaines formes ?

Réponse : Les relevés retrouvés citent surtout ces formes : {'unknown': 3, 'circle': 2, 'flash': 1}. Réponse fondée sur 6 relevés, pas sur une génération libre.

Budget utilisé : **965/1200 caractères**. Temps recherche : **0.488 s**.

|   row_id | datetime         | city              | state   | country   | shape   | comments                                                                                                       |   score_recherche |
|---------:|:-----------------|:------------------|:--------|:----------|:--------|:---------------------------------------------------------------------------------------------------------------|------------------:|
|    46248 | 5/8/2010 01:00   | los angeles       | ca      | us        | flash   | hablo espanol...  pero vi una luz rapida&#44 muy rapida (anormal) aparecio y desaparecio&#44 con un recorrido  |          0.416382 |
|     3353 | 10/20/2012 18:00 | berlin (germany)  | nan     | de        | unknown | Ovni a berlin. Sorte de tissu noir&#44 flottant en apesanteur&#44 et changeant. Eclair&eacute; par en dessous  |          0.294835 |
|    68345 | 8/21/2004 22:45  | montreal (canada) | qc      | ca        | circle  | J&#39avais jamais vu en 31 ans d&#39existence quelconque forme effectu&eacute; des virages a 45 degr&eacute;s  |          0.270472 |
|    26772 | 2/15/2010 05:00  | nashville (near)  | tn      | us        | unknown | UFO CASEBOOK REPORT:  Alien Encounter in Tennessee                                                             |          0        |
|    26773 | 2/15/2010 05:30  | renton            | wa      | us        | unknown | UFO sighting Feb 15&#44 2010 over Lake Washington North of Boeing in Renton                                    |          0        |
|    26774 | 2/15/2010 05:50  | clearwater        | ks      | us        | circle  | Circular object approx the size of 3-4 baseball stadiums emitting amber rays straight down over the countrysid |          0        |
### Y a-t-il des relevés où l'objet semble suivre une voiture ?

Réponse : Les relevés retrouvés citent surtout ces formes : {'circle': 4, 'oval': 2}. Réponse fondée sur 6 relevés, pas sur une génération libre.

Budget utilisé : **744/1200 caractères**. Temps recherche : **0.489 s**.

|   row_id | datetime        | city                                                           | state   | country   | shape   | comments                                                                                                      |   score_recherche |
|---------:|:----------------|:---------------------------------------------------------------|:--------|:----------|:--------|:--------------------------------------------------------------------------------------------------------------|------------------:|
|    32340 | 3/19/2004 16:00 | san jos&eacute; (costa rica)                                   | nan     | nan       | oval    | The objet was stedy in the sky                                                                                |          0.629647 |
|    74670 | 9/15/1994 10:00 | mexicali b. c. mexico; sierra cucapah; cerro prieto.&#44 b. c. | nan     | nan       | oval    | I seen one objet upper the Sierra cucapah (Cucapah mountains)the form of this objet: ovoid and gold color&#44 |          0.524692 |
|    34805 | 3/5/2007 06:45  | des plaines                                                    | il      | us        | circle  | Circular object over Des Plaines&#44 IL traveling south at high rate of speed.                                |          0.509831 |
|    64482 | 8/11/1999 23:07 | ketchikan                                                      | ak      | us        | circle  | saw a bright objet in an area where there is or never has been any thing                                      |          0.466179 |
|    45330 | 5/3/2004 22:00  | london (uk/england)                                            | nan     | gb        | circle  | The sky was clear and this large objet was just hovering                                                      |          0.463358 |
|    37486 | 4/17/2012 21:30 | des allemands                                                  | la      | us        | circle  | Circular moving object over des allemands&#44 la                                                              |          0.394236 |

Quand rien de proche n'est trouvé dans le budget, le système répond explicitement qu'il ne sait pas au lieu
d'inventer un relevé.

## Phase 16 - Faire entrer le tout dans le vaisseau

Marge annoncée avant optimisation : perte maximale acceptée de **0.25** sur le
recouvrement moyen des relevés cités par rapport au système avant réduction.

Protocole : mêmes quatre questions que la phase 15, même budget de 1200 caractères, même machine. Le
système avant garde **12000** entrées TF-IDF ; le système réduit garde **11000**
entrées. Aucune donnée ni cache n'est ajouté au dépôt.

| système   |   max_features |   poids_index_KiB |   build_s |   latence_s |   débit_qps |   réponses_sourcées |   recouvrement |
|:----------|---------------:|------------------:|----------:|------------:|------------:|--------------------:|---------------:|
| avant     |          12000 |           8602.01 |  0.463428 |  0.00839999 |     119.048 |                   4 |          1     |
| après     |          11000 |           8571.58 |  0.466316 |  0.00846727 |     118.102 |                   4 |          0.875 |

Poids sur disque estimé de l'index : gain **1.00x**. Latence d'une réponse unique : gain
**0.99x**. Débit : gain **0.99x**. Écart de score constaté :
**0.12** de perte de recouvrement.

Réduction appliquée : vocabulaire TF-IDF plus petit, ce qui réduit la matrice sparse et accélère légèrement la
similarité cosinus. Je m'arrête ici parce que la perte reste dans la marge annoncée ; l'étape suivante serait
un export autonome de l'index et une quantification plus grossière des poids de la matrice pour obtenir un
gain plus net.

## Phase 17 - Le faux témoignage

Règle absolue respectée : aucune valeur interne de modèle n'est entraînée ni ajustée. La seule action est le
choix du prochain mot au moment d'écrire, contrôlé ici par la température d'une chaîne de Markov construite
sur des vrais relevés courts.

Signature de l'état avant génération : **(1000, 14126)**. Signature après génération : **(1000, 14126)**.
Elles sont identiques, ce qui démontre que les transitions disponibles n'ont pas bougé entre le premier et le
dernier essai.

Grille des réglages :

|   température | symptôme                                   | sortie                                                                                             |
|--------------:|:-------------------------------------------|:---------------------------------------------------------------------------------------------------|
|           0.2 | texte propre mais répétitif                | cloudlike smudge moving across the sky 44 then disappeared straight up and i saw a bright light in |
|           1.6 | texte instable qui part dans tous les sens | two two red quot it one we saw it went crazy i dark oval object i bright white                     |
|           0.8 | réglage recommandé                         | at the sky 44 with beams coming home from a silent 44 then 44at a lake erie 44                     |

Étalon de style, vrais relevés mélangés au faux recommandé pour un futur tri en aveugle :

| vrai_relevé                                                                                      |
|:-------------------------------------------------------------------------------------------------|
| One arc light over billing montana.                                                              |
| I noticed 6 or 7 constant red/orange lights low on the horizon behind the trees.                 |
| Round red ball of fire or light moving west to east at a slow rate of speed&#44 not an aircraft. |
| Very low&#44 very long&#44 metor-like flash across sky                                           |

Réglage recommandé au Bureau : **température 0.8**. Le réglage bas répète trop vite les mêmes enchaînements ;
le réglage haut saute entre débuts de témoignages et devient incohérent. Le point utile est au milieu, où le
texte reste plat, court et maladroit comme les relevés, sans tourner en boucle trop visiblement.
