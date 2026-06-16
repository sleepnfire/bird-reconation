# Guide d'interprétation des entraînements

Ce guide explique chaque métrique, chaque phase de l'entraînement et les actions correctives selon les situations rencontrées.

---

## Table des matières

1. [Sortie console : lire un entraînement en cours](#1-sortie-console--lire-un-entraînement-en-cours)
2. [Les 4 métriques principales](#2-les-4-métriques-principales)
3. [Phases de l'entraînement](#3-phases-de-lentraînement)
4. [Fichiers de sortie](#4-fichiers-de-sortie)
5. [Diagnostic : interpréter les courbes](#5-diagnostic--interpréter-les-courbes)
6. [Rapport de classification](#6-rapport-de-classification)
7. [Matrice de confusion](#7-matrice-de-confusion)
8. [Knowledge Distillation](#8-knowledge-distillation)
9. [QAT (Quantization-Aware Training)](#9-qat-quantization-aware-training)
10. [Hyperparamètres : quand et comment les ajuster](#10-hyperparamètres--quand-et-comment-les-ajuster)
11. [Checklist avant export](#11-checklist-avant-export)

---

## 1. Sortie console : lire un entraînement en cours

Chaque epoch affiche une ligne comme celle-ci :

```
Epoch  12/ 80 — train_loss=1.2345 train_acc=0.6512 — val_loss=1.4567 val_acc=0.5890 — 42s
```

| Champ | Signification |
|-------|---------------|
| `Epoch 12/80` | Epoch actuelle / nombre total. L'epoch est un passage complet sur toutes les images d'entraînement. |
| `train_loss=1.2345` | Loss moyenne sur le jeu d'entraînement pour cette epoch. |
| `train_acc=0.6512` | Proportion d'images correctement classifiées pendant l'entraînement (65,12 %). |
| `val_loss=1.4567` | Loss moyenne sur le jeu de validation (images jamais vues pendant l'entraînement). |
| `val_acc=0.5890` | Accuracy sur le jeu de validation (58,90 %). |
| `42s` | Temps écoulé pour cette epoch. |

**Messages spéciaux :**

| Message | Signification |
|---------|---------------|
| `→ Meilleur modèle sauvegardé (val_loss=X)` | La val_loss vient de battre le meilleur score. Le modèle est sauvegardé. |
| `Backbone dégelé à l'epoch N` | Le backbone (features pré-entraînées) est maintenant entraînable. Attend une augmentation temporaire de la loss. |
| `Early stopping à l'epoch N` | L'entraînement s'arrête car la val_loss n'a pas diminué depuis `patience` epochs. |

---

## 2. Les 4 métriques principales

### 2.1. Loss (train_loss, val_loss)

La **loss** mesure l'erreur du modèle. Plus elle est basse, mieux c'est.

- **CrossEntropyLoss** (par défaut) : mesure la distance entre la distribution prédite et la classe réelle. Avec 558 classes, une loss aléatoire serait `ln(558) ≈ 6.32`.
- **Avec label smoothing** (ε = 0.1) : la loss plancher n'est pas 0 mais l'entropie de la distribution lissée, qui dépend du nombre de classes K : `H ≈ −(1−ε)·ln(1−ε) + ε·ln(K/ε)`. Pour 558 classes : `≈ 0.095 + 0.863 ≈ 0.96`. Une train_loss de 1.2 est donc déjà très bonne avec label smoothing actif.

**Ordres de grandeur typiques (558 classes, MobileNetV2) :**

| Phase | train_loss | val_loss | Commentaire |
|-------|-----------|---------|-------------|
| Epoch 1 (backbone gelé) | 5.0 – 6.0 | 5.5 – 6.5 | Normal : seule la tête apprend |
| Epoch 5 (backbone dégelé) | 3.0 – 4.0 | 3.5 – 4.5 | La loss remonte brièvement, c'est attendu |
| Epoch 20 | 1.5 – 2.5 | 2.0 – 3.0 | Apprentissage en cours |
| Epoch 50+ | 0.5 – 1.5 | 1.5 – 2.5 | Convergence |
| Fin | 0.3 – 0.8 | 1.2 – 2.0 | Un écart train/val de 0.5–1.0 est normal |

### 2.2. Accuracy (train_acc, val_acc)

La **accuracy** est le pourcentage de prédictions correctes. Avec 558 classes, une accuracy aléatoire serait ~0.18 % (1/558).

**Attention pendant l'entraînement :** quand MixUp/CutMix est activé (par défaut), la `train_acc` est calculée sur des images mélangées avec des labels « mous » (soft labels). Elle est donc **systématiquement sous-estimée** par rapport à la réalité. La `val_acc` est toujours calculée sur des images propres et est la métrique fiable.

**Ordres de grandeur typiques :**

| val_acc | Interprétation |
|---------|----------------|
| < 10 % | Le modèle n'apprend pas (vérifier le dataset, le learning rate) |
| 10 – 30 % | Apprentissage démarré mais encore faible |
| 30 – 50 % | Le modèle apprend, bon signe |
| 50 – 70 % | Bon niveau pour 558 classes |
| 70 – 85 % | Très bon, surtout avec des espèces visuellement proches |
| > 85 % | Excellent — vérifier qu'il n'y a pas de fuite de données |

### 2.3. Relation entre les métriques

La loss est plus **sensible** que l'accuracy. Un modèle peut avoir la même accuracy mais une loss très différente :
- `val_loss=1.5, val_acc=70%` : le modèle est assez confiant dans ses prédictions correctes
- `val_loss=3.0, val_acc=70%` : le modèle devine juste mais est incertain (probabilités faibles)

C'est pourquoi le **checkpoint est sauvegardé sur la val_loss** (pas l'accuracy) et l'early stopping surveille aussi la val_loss.

---

## 3. Phases de l'entraînement

L'entraînement se déroule en plusieurs phases distinctes :

### Phase 1 : Backbone gelé (epochs 1 à 3)

```
Epoch   1/ 80 — train_loss=5.8234 train_acc=0.0123 — val_loss=5.2341 val_acc=0.0234 — 38s
Epoch   2/ 80 — train_loss=4.5123 train_acc=0.0567 — val_loss=4.1234 val_acc=0.0789 — 37s
Epoch   3/ 80 — train_loss=3.8901 train_acc=0.1234 — val_loss=3.7654 val_acc=0.1456 — 37s
```

**Ce qui se passe :** Seule la tête de classification (dernière couche) est entraînée. Le backbone (MobileNetV2 pré-entraîné sur ImageNet) est figé. C'est comme apprendre à lire une nouvelle langue en n'utilisant que les structures visuelles qu'on connaît déjà.

**Ce qu'il faut surveiller :**
- La loss doit **descendre régulièrement** dès les premières epochs
- Si elle stagne ou monte → le learning rate de la tête est trop élevé ou trop bas
- L'accuracy est très basse car le backbone ne connaît que les features génériques d'ImageNet, pas encore les oiseaux

**Quand s'inquiéter :** Si après 3 epochs la loss n'a pas bougé, vérifier que `--lr` n'est pas trop petit (essayer `5e-3`).

### Phase 2 : Warmup du learning rate (epochs 1 à 3, simultané)

Le learning rate démarre à 4 % de sa valeur cible et augmente linéairement jusqu'à 100 % en 3 epochs.

```
Epoch 1 : LR tête = 0.00004  (4% de 0.001)
Epoch 2 : LR tête = 0.00052  (52% de 0.001)
Epoch 3 : LR tête = 0.001    (100%)
```

**Pourquoi :** Éviter de détruire les poids pré-entraînés avec un gradient trop fort dès le début. C'est comme préchauffer un moteur.

### Phase 3 : Dégel du backbone (epoch 4)

```
  Backbone dégelé à l'epoch 4
Epoch   4/ 80 — train_loss=3.9500 train_acc=0.1100 — val_loss=3.8200 val_acc=0.1300 — 55s
```

**Ce qui se passe :** Le backbone commence à s'adapter aux oiseaux. Le temps par epoch augmente car il y a beaucoup plus de paramètres à mettre à jour. La loss peut **remonter temporairement** (1-2 epochs) — c'est normal et attendu.

**Le LR différentiel :**
- Backbone : `lr × 0.01` = `0.00001` (très faible pour ne pas « oublier » ImageNet)
- Tête : `lr` = `0.001` (plus fort pour apprendre la classification)

**Ce qu'il faut surveiller :**
- La remontée de loss ne doit pas durer plus de 2-3 epochs
- Si la loss explose → `--backbone-lr-factor` est trop élevé (essayer `0.001`)

### Phase 4 : Entraînement principal (epochs 4 à ~50+)

```
Epoch  20/ 80 — train_loss=1.8234 train_acc=0.4512 — val_loss=2.3456 val_acc=0.3890 — 55s
Epoch  21/ 80 — train_loss=1.7891 train_acc=0.4623 — val_loss=2.2345 val_acc=0.4012 — 54s
  → Meilleur modèle sauvegardé (val_loss=2.2345)
```

**Ce qui se passe :** Le learning rate suit un **cosine decay** (décroît en forme de cosinus de sa valeur max vers ~0). Le modèle affine progressivement sa compréhension.

**Ce qu'il faut surveiller :**
- L'écart entre train_loss et val_loss (voir [diagnostic](#5-diagnostic--interpréter-les-courbes))
- La fréquence des sauvegardes de « meilleur modèle »
- Si aucune sauvegarde depuis 10+ epochs → le modèle stagne

### Phase 5 : Early stopping (optionnel)

```
Early stopping à l'epoch 52 (pas d'amélioration depuis 15 epochs, best_loss=1.8234)
```

**Ce qui se passe :** La val_loss n'a pas diminué depuis `patience` epochs (15 par défaut). L'entraînement s'arrête pour éviter le surapprentissage.

**Ce qu'il faut en conclure :**
- Si early stopping à epoch 20-30 : le modèle apprend vite mais plafonne → essayer plus de données, ou réduire le dropout/regularization
- Si early stopping à epoch 60-70 : entraînement normal, le modèle a bien convergé
- Si pas d'early stopping (80/80 epochs) : on pourrait bénéficier de plus d'epochs (`--epochs 120`)

### Phase 6 : Évaluation finale sur le jeu de test

```
Évaluation finale sur le jeu de test (18732 images)...
Test — loss=1.9234 acc=0.5678
```

Le test set n'est **jamais vu** pendant l'entraînement ni la validation. C'est la mesure non biaisée de la performance du modèle.

- `test_acc ≈ val_acc` : normal, le modèle généralise bien
- `test_acc << val_acc` : problème de distribution entre validation et test (ou surapprentissage sur la validation)
- `test_acc > val_acc` : chanceux ou biais dans le split

---

## 4. Fichiers de sortie

Après l'entraînement, le dossier `output/` contient :

| Fichier | Contenu | Utilisation |
|---------|---------|-------------|
| `best_mobilenetv2.pth` | Checkpoint du meilleur modèle (val_loss minimale) | **Le modèle à utiliser** pour l'export et le déploiement |
| `last_mobilenetv2.pth` | Checkpoint du dernier epoch | Utile pour reprendre l'entraînement (`--resume`) |
| `training_history.json` | Historique complet : 4 métriques par epoch | Analyser l'entraînement après coup |
| `training_curves.png` | Graphiques loss et accuracy | Diagnostic visuel rapide |
| `classification_report.json` | Precision/recall/F1 par espèce | Identifier les espèces problématiques |
| `confusion_matrix.npy` | Matrice de confusion (numpy) | Analyse avancée |
| `confusion_matrix.png` | Heatmap de la matrice (si ≤ 50 classes) | Voir quelles espèces sont confondues |

---

## 5. Diagnostic : interpréter les courbes

### 5.1. Entraînement sain

```
train_loss  ████████████████░░░░░░░░  ↘ descend régulièrement
val_loss    ████████████████████░░░░  ↘ descend aussi, légèrement au-dessus
```

**Signes :**
- train_loss et val_loss descendent ensemble
- Écart train/val stable et modéré (0.3 – 1.0)
- val_acc augmente régulièrement
- **Action :** Tout va bien, laisser tourner.

### 5.2. Overfitting (surapprentissage)

```
train_loss  ████████████░░░░░░░░░░░░  ↘ continue de descendre
val_loss    ████████████████████████  ↗ remonte après un point bas
```

**Signes :**
- train_loss descend, val_loss remonte
- train_acc > 90 %, val_acc stagne à 50-60 %
- Grand écart train/val (> 1.5)

**Le modèle « apprend par coeur » les images d'entraînement** au lieu d'apprendre des patterns généraux.

**Actions correctives (par ordre de priorité) :**

| Action | Paramètre | Pourquoi |
|--------|-----------|----------|
| Ajouter des données | Plus d'images par espèce | Le plus efficace contre l'overfitting |
| Augmenter le dropout | `--dropout 0.6` ou `0.7` | Force le modèle à ne pas se fier à un seul neurone |
| Augmenter le label smoothing | `--label-smoothing 0.2` | Empêche le modèle d'être trop confiant |
| Augmenter la data augmentation | `--randaugment-magnitude 15` | Plus de variété artificielle |
| Activer MixUp/CutMix | Ne pas mettre `--no-mixup` | Régularisation forte |
| Augmenter le weight decay | `--weight-decay 5e-2` | Pénalise les poids trop grands |
| Réduire la capacité du modèle | Utiliser MobileNetV2 au lieu d'EfficientNet | Moins de paramètres = moins d'overfitting |

### 5.3. Underfitting (sous-apprentissage)

```
train_loss  ████████████████████████  → stagne à un niveau élevé
val_loss    ████████████████████████  → stagne aussi, écart faible
```

**Signes :**
- train_loss et val_loss stagnent à un niveau élevé
- train_acc et val_acc sont proches mais basses (< 30 %)
- L'écart train/val est faible (< 0.3)

**Le modèle n'a pas assez de capacité** pour apprendre le problème, ou le learning rate est inadapté.

**Actions correctives :**

| Action | Paramètre | Pourquoi |
|--------|-----------|----------|
| Augmenter le LR | `--lr 3e-3` ou `5e-3` | Le modèle apprend trop lentement |
| Réduire le dropout | `--dropout 0.3` | Trop de régularisation bride le modèle |
| Réduire le label smoothing | `--label-smoothing 0.05` | Idem |
| Désactiver MixUp | `--no-mixup` | MixUp rend l'apprentissage plus dur |
| Utiliser un modèle plus grand | `--model efficientnet_b0` | Plus de capacité |
| Plus d'epochs | `--epochs 120` | Le modèle a besoin de plus de temps |
| Dégeler le backbone plus tôt | `--freeze-backbone-epochs 1` | Le backbone s'adapte plus tôt |

### 5.4. Learning rate trop élevé

```
train_loss  ████████████████████████  → oscille fortement / explose
val_loss    ████████████████████████  → idem
```

**Signes :**
- La loss oscille d'une epoch à l'autre (ex : 2.3 → 3.1 → 2.5 → 3.4)
- Ou la loss explose (NaN ou valeurs > 10)

**Actions :**
- Réduire `--lr` d'un facteur 3 à 10 (ex : `1e-3` → `3e-4`)
- Augmenter `--warmup-epochs` à 5-10
- Activer le gradient clipping `--clip-grad 1.0` (déjà actif par défaut)

### 5.5. Learning rate trop bas

```
train_loss  ████████████████████████  → descend extrêmement lentement
val_loss    ████████████████████████  → suit de très près
```

**Signes :**
- Pas d'overfitting, mais progrès très lents
- Après 30 epochs, les métriques ont à peine bougé
- **Action :** Augmenter `--lr` (ex : `1e-3` → `3e-3`)

---

## 6. Rapport de classification

Le fichier `classification_report.json` donne 3 métriques **par espèce** :

### Precision

```
precision = vrais positifs / (vrais positifs + faux positifs)
```

**En français :** « Quand le modèle dit que c'est un rouge-gorge, a-t-il raison ? »

- **Precision = 0.95** : sur 100 fois qu'il dit « rouge-gorge », il a raison 95 fois
- **Precision basse** (< 0.5) : le modèle confond d'autres espèces avec celle-ci

### Recall (rappel)

```
recall = vrais positifs / (vrais positifs + faux négatifs)
```

**En français :** « Parmi tous les vrais rouges-gorges, combien sont correctement identifiés ? »

- **Recall = 0.90** : sur 100 vrais rouges-gorges, le modèle en reconnaît 90
- **Recall bas** (< 0.5) : le modèle rate souvent cette espèce (la classe dans une autre)

### F1-score

```
F1 = 2 × (precision × recall) / (precision + recall)
```

**En français :** Moyenne harmonique de precision et recall. C'est **la métrique unique la plus utile** par espèce.

- **F1 > 0.8** : l'espèce est bien reconnue
- **F1 entre 0.5 et 0.8** : correcte mais perfectible
- **F1 < 0.5** : problématique — investiguer (peu de données ? espèces visuellement proches ?)
- **F1 = 0** : l'espèce n'est jamais correctement identifiée

### Support

Le nombre d'images de test pour cette espèce. Un F1 calculé sur 3 images n'est pas fiable. Minimum ~20 images pour être significatif.

### Moyennes globales

| Métrique | Signification |
|----------|---------------|
| `macro avg` | Moyenne non pondérée de toutes les classes. Donne autant d'importance à une espèce rare qu'à une espèce commune. |
| `weighted avg` | Moyenne pondérée par le support (nombre d'images). Reflète la performance « réelle ». |

**Que regarder en priorité :**

1. `weighted avg F1` → performance globale du modèle
2. Les espèces avec F1 < 0.5 → ce sont vos cas problématiques
3. Les espèces avec un support < 10 → données insuffisantes, résultats non fiables

### Exemple de lecture

```json
{
  "erithacus_rubecula": {
    "precision": 0.92,
    "recall": 0.88,
    "f1-score": 0.90,
    "support": 45
  },
  "parus_major": {
    "precision": 0.45,
    "recall": 0.78,
    "f1-score": 0.57,
    "support": 38
  }
}
```

**Interprétation :**
- **Rouge-gorge** (F1=0.90) : très bien reconnu. Precision et recall élevés.
- **Mésange charbonnière** (F1=0.57) : precision basse (0.45) = le modèle la confond avec d'autres espèces. Recall correct (0.78) = il la détecte quand même. → Regarder la matrice de confusion pour trouver avec quelle espèce elle est confondue (probablement mésange bleue ou mésange noire).

---

## 7. Matrice de confusion

### Lecture de la matrice

La matrice de confusion est un tableau NxN (N = nombre de classes).

```
              Prédiction →
              Rouge-gorge  Mésange  Merle
Réalité ↓
Rouge-gorge       42          2       1
Mésange            8         30       0
Merle              1          0      47
```

- **Diagonale** (42, 30, 47) : prédictions correctes. Plus c'est élevé, mieux c'est.
- **Hors diagonale** : erreurs. Ligne = vraie classe, colonne = ce que le modèle a prédit.
- `Mésange → Rouge-gorge = 8` : 8 mésanges ont été classées comme rouges-gorges.

### Que chercher

1. **Cases très sombres hors diagonale** : confusions fréquentes entre deux espèces
2. **Lignes avec diagonale faible** : espèces souvent mal classifiées
3. **Colonnes avec beaucoup de valeurs hors diagonale** : espèces « attracteurs » (le modèle classe trop de choses dans cette catégorie)

### Actions selon les confusions observées

| Observation | Cause probable | Action |
|-------------|---------------|--------|
| Deux espèces se confondent mutuellement | Visuellement très similaires (ex : mésange bleue / mésange azurée) | Ajouter des images de meilleure qualité, crop sur les détails distinctifs |
| Une espèce absorbe les autres | Classe surreprésentée | Vérifier l'équilibre du dataset, activer le WeightedRandomSampler (déjà actif) |
| Confusion unidirectionnelle (A→B mais pas B→A) | Peu d'images de A | Ajouter plus d'images de A |
| Beaucoup de confusions entre espèces d'une même famille | Features discriminantes trop subtiles | Essayer un modèle plus grand (EfficientNet-B0), ou utiliser la knowledge distillation |

---

## 8. Knowledge Distillation

La distillation est un entraînement spécial où un **gros modèle** (teacher, ViT-B/16) guide un **petit modèle** (student, MobileNetV2).

### Métriques spécifiques

La ligne d'entraînement est identique à un entraînement normal. La différence est dans la loss :

```
loss = alpha × KL_soft + (1 - alpha) × hard_loss
```

| Composante | Par défaut | Signification |
|------------|-----------|---------------|
| `KL_soft` | pondération 0.7 | Distance entre la distribution du student et celle du teacher (soft targets) |
| `hard_loss` | pondération 0.3 | CrossEntropy classique avec les vrais labels |
| `T` (température) | 4.0 | Plus T est élevé, plus les soft targets sont « doux » (information inter-classes) |

### Interpréter la loss de distillation

La train_loss en distillation est **plus élevée** qu'en entraînement normal. C'est normal : la loss KL sur les soft targets est plus dure à minimiser.

**Ce qu'il faut surveiller :**
- La **val_loss et val_acc** (calculées normalement, sans distillation) sont les métriques fiables
- Comparer la `val_acc` du student distillé vs. un student entraîné sans distillation
- Un student distillé devrait atteindre 2-5 % de val_acc en plus qu'un student normal

### Quand la distillation ne fonctionne pas

| Symptôme | Cause | Action |
|----------|-------|--------|
| Student distillé pire que student normal | Teacher trop mauvais | Entraîner un meilleur teacher |
| Student ne converge pas | T trop élevé | Réduire `--distill-temperature` à 2.0 |
| Student copie le teacher sans le dépasser | alpha trop élevé | Réduire `--distill-alpha` à 0.5 (plus de poids sur les hard labels) |

---

## 9. QAT (Quantization-Aware Training)

Le QAT est une phase **post-entraînement** qui prépare le modèle à tourner en INT8 (8 bits au lieu de 32 bits) sur les accélérateurs matériels.

### Lecture de la sortie

```
QAT   1/  5 — train_loss=0.8234 train_acc=0.7512 — val_loss=1.0567 val_acc=0.7290 — 65s
QAT   2/  5 — train_loss=0.7891 train_acc=0.7623 — val_loss=1.0345 val_acc=0.7312 — 64s
```

**Ce qu'il faut surveiller :**
- La **val_acc du QAT** par rapport à la val_acc finale de l'entraînement normal
- Perte acceptable : 0.5-2 % d'accuracy (ex : 73 % → 71 %)
- Perte inquiétante : > 3 % d'accuracy

### Impact attendu

| Métrique | Avant QAT (float32) | Après QAT (INT8) | Acceptable |
|----------|---------------------|-------------------|-----------|
| val_acc | 73.0 % | 71.0 – 72.5 % | Oui |
| val_acc | 73.0 % | 68.0 % | Non — essayer plus d'epochs QAT |
| Taille modèle | ~13 Mo | ~3.5 Mo | Attendu (÷4 environ) |
| Vitesse inférence | Baseline | 2-4× plus rapide | Sur accélérateur INT8 |

### Si la perte de précision est trop grande

| Action | Paramètre | Effet |
|--------|-----------|-------|
| Plus d'epochs QAT | `--qat-epochs 10` | Le modèle s'adapte mieux à la quantification |
| Réduire le LR QAT | Modifier dans le code (actuellement `lr × 0.1`) | Ajustement plus fin |

---

## 10. Hyperparamètres : quand et comment les ajuster

### Tableau de diagnostic rapide

| Problème observé | 1er paramètre à toucher | Valeur à essayer |
|-----------------|------------------------|-----------------|
| Loss ne descend pas du tout | `--lr` | `3e-3` ou `5e-3` |
| Loss oscille fortement | `--lr` | `3e-4` ou `1e-4` |
| Loss explose (NaN) | `--lr` + `--clip-grad` | `1e-4` + `1.0` |
| Overfitting sévère | `--dropout` | `0.6` ou `0.7` |
| Overfitting modéré | `--label-smoothing` | `0.15` ou `0.2` |
| Underfitting | `--dropout` | `0.3` |
| Early stopping trop tôt (< epoch 25) | `--patience` | `20` ou `25` |
| Entraînement trop long sans progrès | `--patience` | `10` |
| Images de mauvaise qualité / peu nombreuses | `--randaugment-magnitude` | `15` ou `18` |
| Accuracy correcte mais loss élevée | `--focal-loss` | Active |
| Classes rares ratées | `--focal-loss` + `--focal-gamma` | Active + `3.0` |

### Ordre recommandé d'expérimentation

1. **Baseline** : lancer avec les paramètres par défaut
2. **Learning rate** : si les résultats ne sont pas satisfaisants, c'est le premier levier
3. **Regularisation** (dropout, label smoothing, weight decay) : ajuster l'écart train/val
4. **Data augmentation** : si peu de données
5. **Architecture** : changer de modèle en dernier recours (ne change rien si le problème est dans les données)

### Correspondance modèle / cible matérielle

| Modèle | Paramètres | Cible | `--image-size` |
|--------|-----------|-------|----------------|
| MobileNetV2 | 3.5M | Pi Zero + IMX500 (≤ 8 Mo INT8) | 224 |
| EfficientNet-B0 | 5.3M | Pi Zero + IMX500 (≤ 8 Mo INT8) | 224 |
| EfficientNetV2-B2 | 10.1M | Pi Zero + IMX500 | 260 |
| ViT-B/16 | 86.6M | Pi 5 + Hailo-10H (83.6%) + Teacher distillation | 224 |

---

## 11. Checklist avant export

Avant de lancer `export.py`, vérifier ces points :

### Performance minimale attendue

| Cible | Modèle | val_acc minimale | val_acc objectif |
|-------|--------|-----------------|-----------------|
| 558 classes (Europe) | MobileNetV2 | > 40 % | 55-65 % |
| 558 classes (Europe) | EfficientNet-B0 | > 50 % | 65-75 % |
| ~20 classes (jardin) | MobileNetV2 | > 80 % | 90-95 % |
| ~20 classes (jardin) | EfficientNet-B0 | > 85 % | 92-97 % |

### Points de contrôle

- [ ] `val_loss` a atteint un plateau (early stopping ou fin des epochs)
- [ ] `val_acc` est dans la plage attendue pour le nombre de classes
- [ ] L'écart `train_acc - val_acc` est < 15 points (pas de surapprentissage majeur)
- [ ] Le rapport de classification ne montre pas d'espèces avec F1 = 0
- [ ] La matrice de confusion ne montre pas de confusions systématiques graves
- [ ] Si QAT : la perte d'accuracy est < 3 %

### Export

```bash
# Vérifier la taille pour IMX500
python export.py --checkpoint output/best_mobilenetv2.pth --target onnx --check-size

# Pour le Hailo
python export.py --checkpoint output/best_efficientnet_b0.pth --target hailo

# Pour l'IMX500 (quantification statique via Sony MCT)
python export.py --checkpoint output/best_mobilenetv2.pth --target imx500 --dataset dataset/europe
```

---

## Glossaire rapide

| Terme | Définition |
|-------|------------|
| **Epoch** | Un passage complet sur toutes les images d'entraînement |
| **Batch** | Groupe d'images traitées ensemble (taille par défaut : 32) |
| **Loss** | Score d'erreur du modèle (plus c'est bas, mieux c'est) |
| **Accuracy** | % de prédictions correctes |
| **Backbone** | Partie pré-entraînée du réseau (features visuelles) |
| **Tête** | Dernière couche qui fait la classification |
| **Fine-tuning** | Adapter un modèle pré-entraîné à une nouvelle tâche |
| **Warmup** | Montée progressive du learning rate au départ |
| **Cosine decay** | Décroissance du learning rate en forme de cosinus |
| **Label smoothing** | Adoucir les labels (0.9 au lieu de 1.0) pour réduire la surconfiance |
| **MixUp** | Mélanger deux images et leurs labels (régularisation) |
| **CutMix** | Coller un patch d'une image sur une autre (régularisation) |
| **EMA** | Moyenne glissante des poids du modèle (stabilise les prédictions) |
| **Focal Loss** | Loss qui insiste sur les exemples difficiles |
| **QAT** | Entraîner en simulant la quantification INT8 |
| **Early stopping** | Arrêter quand le modèle ne s'améliore plus |
| **Overfitting** | Le modèle apprend par coeur au lieu de généraliser |
| **Underfitting** | Le modèle n'a pas assez appris |
| **Distillation** | Un petit modèle apprend d'un gros modèle |
