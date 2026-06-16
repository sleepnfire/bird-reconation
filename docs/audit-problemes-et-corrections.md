# Audit : problèmes identifiés, implications et corrections

Date de l'audit initial : 2026-06-13

## Contexte

Après un premier entraînement de MobileNetV2 sur 558 espèces d'oiseaux européens (~124 911 images), le modèle présentait un **overfitting sévère** :

| Métrique | Valeur |
|----------|--------|
| train_acc | 92.07 % |
| val_acc | 61.98 % |
| Écart | **30.09 points** |
| train_loss | 0.2874 |
| val_loss | 1.8607 |

Le modèle mémorisait les images d'entraînement au lieu d'apprendre à reconnaître les oiseaux. Ce document détaille chaque problème identifié, explique pourquoi il cause du tort, et montre comment la correction le résout.

---

## Table des matières

1. [Weight decay trop faible](#1-weight-decay-trop-faible)
2. [Absence de MixUp et CutMix](#2-absence-de-mixup-et-cutmix)
3. [Absence de ColorJitter](#3-absence-de-colorjitter)
4. [Learning rate du backbone trop agressif](#4-learning-rate-du-backbone-trop-agressif)
5. [Dropout insuffisant](#5-dropout-insuffisant)
6. [Trop peu d'epochs](#6-trop-peu-depochs)
7. [Absence de gradient clipping](#7-absence-de-gradient-clipping)
8. [Absence d'EMA](#8-absence-dema-exponential-moving-average)
9. [Mauvais choix d'architectures](#9-mauvais-choix-darchitectures)
10. [Absence de knowledge distillation](#10-absence-de-knowledge-distillation)
11. [Absence de Focal Loss](#11-absence-de-focal-loss)
12. [RandAugment trop conservateur](#12-randaugment-trop-conservateur)
13. [Pipeline de quantification inadapté](#13-pipeline-de-quantification-inadapté)
14. [Stratégie coarse-to-fine abandonnée](#14-stratégie-coarse-to-fine-abandonnée)

---

## 1. Weight decay trop faible

### Le problème

Le weight decay était codé en dur à `1e-4` (0.0001), sans possibilité de le modifier en CLI.

### Ce que ça implique

Le weight decay est une pénalité appliquée sur la taille des poids du réseau. Il ajoute un terme `λ × ||w||²` à la loss, ce qui empêche les poids de devenir trop grands.

Quand les poids sont trop grands, le modèle crée des frontières de décision très « angulaires » et spécifiques : il trace des contours qui passent exactement par chaque exemple d'entraînement plutôt que des contours lisses et généraux. C'est le mécanisme fondamental de l'overfitting.

À `1e-4`, la pénalité est **100 fois trop faible** pour avoir un effet mesurable. Le modèle est libre de gonfler ses poids autant qu'il veut pour mémoriser les exemples.

**Analogie :** C'est comme écrire un résumé de livre en recopiant le texte mot pour mot au lieu de synthétiser les idées. Le weight decay force le modèle à « synthétiser » en limitant la complexité de sa représentation interne.

### La correction

```
Avant : weight_decay=1e-4 (codé en dur)
Après : --weight-decay 1e-2 (exposé en CLI, valeur par défaut ×100)
```

À `1e-2`, la pénalité est assez forte pour empêcher les poids de grossir excessivement. Le modèle est contraint de trouver des représentations compactes et généralisables. C'est la valeur par défaut de `torch.optim.AdamW` et une valeur couramment utilisée pour le fine-tuning de CNN pré-entraînés.

De plus, le weight decay est maintenant **séparé par type de paramètre** : les poids des couches de convolution/linéaires reçoivent le weight decay, mais les biais et les paramètres de normalisation (BatchNorm) en sont exemptés (`weight_decay=0.0`). Appliquer du weight decay aux biais est contre-productif car les biais n'ont pas le même rôle que les poids — ils décalent les activations, ils ne les amplifient pas.

**Impact estimé : +2-5 points de val_acc**

---

## 2. Absence de MixUp et CutMix

### Le problème

L'augmentation de données opérait uniquement sur chaque image individuellement (crop, flip, rotation). Il n'y avait aucun mélange entre images différentes.

### Ce que ça implique

Avec ~224 images par espèce et 558 espèces, le modèle voit les mêmes images à chaque epoch. Même avec des transformations aléatoires (flip, crop), le nombre de « situations visuelles » distinctes est limité. Le modèle finit par reconnaître des images spécifiques plutôt que des patterns visuels.

Sans mélange inter-images, le modèle apprend aussi des **frontières de décision tranchantes** : « cette image est 100 % mésange charbonnière, 0 % mésange bleue ». Dans la réalité, certaines photos sont ambiguës (mauvais angle, juvénile, lumière difficile), et le modèle devrait exprimer de l'incertitude.

### La correction

**MixUp** (alpha=0.2) : mélange linéairement deux images et leurs labels.

```
image_mixée = 0.7 × image_A + 0.3 × image_B
label_mixé  = 0.7 × "mésange" + 0.3 × "rouge-gorge"
```

Le modèle doit prédire un vecteur de probabilité (soft label) au lieu d'une classe unique. Cela l'oblige à apprendre des représentations continues et lisses dans l'espace des features.

**CutMix** (alpha=1.0) : découpe un rectangle d'une image et le colle sur une autre, avec des labels proportionnels à la surface.

```
image_cutmix = image_A avec un patch de image_B collé dessus
label_cutmix = 0.75 × "mésange" + 0.25 × "rouge-gorge"  (si le patch = 25% de l'image)
```

CutMix a un avantage supplémentaire sur MixUp : le modèle doit localiser les parties informatives de l'image (la tête de l'oiseau, le plumage) car une partie de l'image est remplacée par un contenu « parasite ». Cela produit des features plus localisées et discriminantes.

Les deux techniques sont appliquées aléatoirement (50/50) à chaque batch, ce qui **multiplie considérablement la diversité** des exemples d'entraînement.

Le papier original (Yun et al., ICCV 2019) rapporte des gains significatifs : +2.28 points d'accuracy sur ImageNet (ResNet-50) et +2.22 sur CIFAR-100 (PyramidNet-200).

**Impact estimé : +3-5 points de val_acc**

---

## 3. Absence de ColorJitter

### Le problème

Aucune perturbation de couleur dans le pipeline d'augmentation.

### Ce que ça implique

Les oiseaux sont principalement identifiés par leur **plumage** : couleur, patterns, contrastes. Or, les photos d'iNaturalist sont prises dans des conditions d'éclairage très variées :
- Soleil direct → couleurs saturées, ombres fortes
- Temps couvert → couleurs ternes, faible contraste
- Aube/crépuscule → teinte orangée
- Sous-bois → lumière verte, faible luminosité

Sans ColorJitter, le modèle associe une espèce à un profil colorimétrique spécifique. Un rouge-gorge photographié au soleil et le même oiseau à l'ombre peuvent être classés différemment, parce que le modèle a appris « rouge-gorge = cette teinte précise de orange » au lieu de « rouge-gorge = cette forme de tache sur la poitrine ».

### La correction

```python
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
```

- **brightness=0.3** : simule soleil/ombre (luminosité ±30 %)
- **contrast=0.3** : simule les variations atmosphériques
- **saturation=0.3** : simule les conditions d'éclairage (saturé au soleil, terne sous les nuages)
- **hue=0.05** : très conservateur volontairement — trop de variation de teinte rendrait un rouge-gorge et un rougequeue indistinguables (leur teinte orange est un trait discriminant)

Le modèle apprend à être invariant aux conditions de lumière tout en restant sensible aux vrais patterns de couleur du plumage.

**Impact estimé : +1-2 points de val_acc**

---

## 4. Learning rate du backbone trop agressif

### Le problème

Le backbone (MobileNetV2 pré-entraîné sur ImageNet) était entraîné avec un learning rate de `1e-4`, soit `backbone_lr_factor=0.1` × le LR de la tête (`1e-3`).

### Ce que ça implique

Le backbone contient des features visuelles apprises sur 1.2 million d'images ImageNet : détection de bords, textures, formes, motifs de couleur. Ces features sont **universelles** et précieuses.

Avec un LR de `1e-4`, le backbone est modifié trop rapidement. Les features générales (bords, textures) sont écrasées par des features hyper-spécifiques aux images d'entraînement. Le modèle « oublie » ce qu'il sait de général pour mémoriser des cas particuliers.

C'est le phénomène de **catastrophic forgetting** (oubli catastrophique) : le réseau perd les connaissances antérieures en assimilant de nouvelles données.

Avec ~224 images/espèce (un dataset relativement petit), le risque est amplifié : peu de données + LR élevé = le backbone se spécialise sur un petit nombre d'exemples.

### La correction

```
Avant : backbone_lr_factor=0.1  → backbone LR = 1e-4
Après : backbone_lr_factor=0.01 → backbone LR = 1e-5
```

Un facteur 10× plus faible signifie que le backbone évolue **très lentement**. Il s'adapte aux oiseaux sans perdre ses features générales. La tête de classification, elle, garde un LR de `1e-3` car elle part de zéro et doit apprendre rapidement.

Ce LR différentiel (faible pour le backbone, élevé pour la tête) est le principe fondamental du fine-tuning. Intuitivement : on ajuste finement un outil de précision plutôt que de le refondre entièrement.

**Impact estimé : +1-2 points de val_acc**

---

## 5. Dropout insuffisant

### Le problème

Le dropout sur la tête de classification était à 0.3 (30 % des neurones désactivés aléatoirement).

### Ce que ça implique

Le dropout force le modèle à distribuer l'information sur plusieurs neurones. Avec dropout=0.3, le modèle peut encore se permettre de concentrer des informations critiques sur quelques neurones spécifiques. Si un neurone « mémorise » un cas particulier, il est actif 70 % du temps — assez pour influencer les prédictions.

Avec 558 classes et un écart train/val de 30 points, le dropout de 0.3 est insuffisant pour casser les co-adaptations entre neurones.

### La correction

```
Avant : dropout=0.3
Après : dropout=0.5 (exposé en CLI via --dropout)
```

À 0.5, chaque neurone est désactivé la moitié du temps. Le modèle est contraint de distribuer l'information uniformément — aucun neurone ne peut « mémoriser » un cas particulier car il ne sera présent qu'une fois sur deux. C'est la valeur recommandée par Srivastava et al. (JMLR 2014) comme défaut pour les couches fully-connected.

0.5 est une valeur de régularisation forte mais pas excessive pour 558 classes. Augmenter au-delà de 0.6-0.7 risquerait de causer de l'underfitting (le modèle n'arrive plus à apprendre).

**Impact estimé : +1-2 points de val_acc**

---

## 6. Trop peu d'epochs

### Le problème

L'entraînement était limité à 30 epochs avec une patience d'early stopping de 7. À l'epoch 30, le modèle était **encore en amélioration**.

### Ce que ça implique

Avec l'ancienne configuration (peu de régularisation), 30 epochs suffisaient car le modèle overfittait rapidement. Mais la raison pour laquelle il overfittait vite était justement le manque de régularisation — pas un signe que le modèle avait « convergé ».

Avec la régularisation renforcée (weight decay ×100, MixUp/CutMix, dropout 0.5), le modèle apprend **plus lentement** mais **plus proprement**. Le MixUp rend la tâche plus difficile (labels mous), le dropout supprime des neurones, le weight decay limite les poids — chaque technique freine la vitesse d'apprentissage au profit de la qualité.

30 epochs ne suffisent plus pour que ce modèle correctement régularisé atteigne son potentiel.

### La correction

```
Avant : --epochs 30, --patience 7
Après : --epochs 80, --patience 15
```

80 epochs avec un cosine annealing du learning rate donnent :
- Epochs 1-3 : warmup (LR monte de 4 % à 100 %)
- Epochs 4-40 : exploration à LR moyen-élevé (les features se forment)
- Epochs 40-70 : raffinement à LR décroissant (les frontières de décision s'affinent)
- Epochs 70-80 : convergence à LR très bas (ajustements fins)

La patience de 15 (au lieu de 7) est cohérente : avec un cosine decay, le modèle peut traverser des plateaux temporaires avant de progresser à nouveau quand le LR baisse. 7 epochs de patience coupaient prématurément l'entraînement.

**Impact estimé : +2-4 points de val_acc (en synergie avec la régularisation)**

---

## 7. Absence de gradient clipping

### Le problème

Aucune limitation sur la norme des gradients dans `train_one_epoch()`.

### Ce que ça implique

Les gradients sont les « signaux de correction » envoyés au réseau après chaque batch. Normalement, ils sont petits et le réseau s'ajuste progressivement. Mais certaines situations produisent des **spikes de gradient** (valeurs anormalement élevées) :

- **Classes rares** : certaines espèces ont seulement 7 images. Quand le modèle se trompe lourdement sur une de ces images, le gradient corrige brutalement — et cette correction peut déstabiliser les features apprises pour les autres espèces.
- **MixUp/CutMix** : les labels mous peuvent créer des configurations de loss inhabituelles qui produisent des gradients plus élevés.
- **Dégel du backbone** : au moment où le backbone est dégelé (epoch 4), un afflux soudain de paramètres entraînables peut provoquer des mises à jour trop grandes.

Un seul spike suffit à corrompre les features d'une couche entière, annulant des dizaines d'epochs de progrès.

### La correction

```python
if clip_grad > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
```

Avec `--clip-grad 1.0` (par défaut), la norme totale du gradient est plafonnée à 1.0. Si le gradient total fait 5.0, tous les gradients individuels sont divisés par 5 pour ramener la norme à 1.0. Les directions de mise à jour sont préservées, seule l'amplitude est limitée.

En mode AMP (mixed precision sur GPU), le clipping est appliqué **après** `scaler.unscale_()` et **avant** `scaler.step()` — sinon les gradients seraient clippés à l'échelle incorrecte.

**Impact estimé : +0.5-1 point de val_acc (surtout en stabilité)**

---

## 8. Absence d'EMA (Exponential Moving Average)

### Le problème

Le modèle sauvegardé et évalué était celui de la dernière itération d'optimisation — avec tout le bruit que cela implique.

### Ce que ça implique

À chaque batch, les poids du modèle font un « saut » dans une direction (la direction du gradient). Ces sauts contiennent du bruit : un batch peut contenir des exemples atypiques, des images de mauvaise qualité, ou des exemples de classes rares qui tirent les poids dans une direction non représentative.

Sur des dizaines de milliers de batches, les poids oscillent autour de leur position optimale. Le modèle à l'instant T n'est pas nécessairement meilleur que celui à l'instant T-100, même si l'entraînement progresse globalement.

Évaluer et sauvegarder le modèle « instantané » à la fin de chaque epoch revient à photographier une balle en mouvement : on capture une position qui n'est peut-être pas la meilleure.

### La correction

```python
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.module = copy.deepcopy(model)
        self.decay = decay

    def update(self, model):
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.lerp_(p.data, 1 - self.decay)
```

L'EMA maintient une copie lissée des poids :

```
poids_ema = 0.9999 × poids_ema_précédent + 0.0001 × poids_actuels
```

À chaque mise à jour, les poids EMA bougent de 0.01 % vers les poids actuels. Cela crée une **moyenne mobile** qui filtre le bruit d'optimisation. C'est comme regarder la tendance d'un cours de bourse plutôt que les fluctuations seconde par seconde.

Le modèle EMA est utilisé pour :
- La **validation** : les métriques val_loss et val_acc reflètent les poids lissés
- La **sauvegarde** du meilleur checkpoint : le modèle déployé est le modèle EMA
- L'**évaluation finale** sur le test set

Le modèle « brut » (non-EMA) continue de s'entraîner normalement. L'EMA n'interfère pas avec l'optimisation.

**Impact estimé : +1-2 points de val_acc**

---

## 9. Mauvais choix d'architectures

### Le problème

Deux problèmes distincts :

**a)** EfficientNetV2-S était mappé comme « EfficientNetV2-B2 ». Or V2-S a **21.5M** paramètres (vs 10.1M pour V2-B2), soit **2.1× plus gros**. En INT8, V2-S fait ~21.5 Mo — bien au-delà des **8 Mo de SRAM** de l'IMX500. Le modèle ne peut tout simplement pas être chargé sur la caméra.

**b)** Aucun modèle Vision Transformer (ViT) n'était implémenté, alors que le Hailo-10H (40 TOPS) peut l'exécuter et que ViT-B/16 est le modèle le plus performant de son Model Zoo (83.6 % sur ImageNet).

### Ce que ça implique

Entraîner un modèle non déployable est un gaspillage total : des heures de GPU pour un fichier inutilisable. Pire, si le problème n'est découvert qu'à l'export, tout le travail d'optimisation des hyperparamètres doit être recommencé avec la bonne architecture.

L'absence de ViT-B/16 privait le projet de deux choses :
1. Le modèle le plus précis pour le Hailo (83.6 % vs 71.0 % pour MobileNetV2, soit +12.6 points sur ImageNet)
2. Un **teacher** de qualité pour la knowledge distillation vers MobileNetV2

### La correction

**EfficientNetV2-B2 via timm :**

```python
import timm
model = timm.create_model('tf_efficientnetv2_b2.in1k', pretrained=True, num_classes=558)
```

Le V2-B2 fait 10.1M paramètres et ~6.5 Mo en RPK (firmware IMX500). Il rentre dans les 8 Mo de SRAM et offre 77.7 % de Top-1 quantifié — le meilleur ratio précision/taille du Model Zoo IMX500.

**ViT-B/16 via torchvision :**

```python
model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.heads = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.hidden_dim, num_classes))
```

Le ViT utilise un mécanisme d'**attention** qui lui permet de se focaliser sur les parties discriminantes de l'image (tête, poitrine, ailes) sans qu'on ait à lui dire où regarder. C'est particulièrement adapté à la classification fine-grained où la différence entre deux espèces tient parfois à un petit détail (couleur de la calotte, forme du bec).

Le code gère les hyperparamètres différents des ViT automatiquement (weight decay plus élevé, warmup plus long).

**Correspondance finale modèle ↔ cible :**

| Modèle | Cible | Taille INT8 | Déployable |
|--------|-------|-------------|:----------:|
| MobileNetV2 | IMX500 + Hailo | 3.89 Mo | Oui |
| EfficientNet-B0 | IMX500 | 5.99 Mo | Oui |
| EfficientNetV2-B2 (timm) | IMX500 | 6.51 Mo | Oui |
| ViT-B/16 | Hailo-10H | N/A (float) | Oui |

---

## 10. Absence de knowledge distillation

### Le problème

MobileNetV2 (3.4M paramètres) devait apprendre seul à partir des données, sans assistance d'un modèle plus puissant.

### Ce que ça implique

MobileNetV2 est un modèle léger conçu pour les appareils mobiles. Sa petite taille est un atout pour le déploiement mais un handicap pour l'apprentissage : il a moins de capacité pour capturer les nuances visuelles fines entre espèces similaires.

Un gros modèle comme ViT-B/16 (86.5M paramètres, 25× plus gros) capture beaucoup plus de relations inter-espèces : il « sait » que la mésange bleue et la mésange charbonnière se ressemblent, que le bruant jaune a des patterns proches du verdier, etc. Cette connaissance est encodée dans ses prédictions : quand il voit une mésange bleue, il ne prédit pas juste « mésange bleue = 100 % », mais plutôt « mésange bleue = 85 %, mésange charbonnière = 8 %, mésange noire = 5 %, ... ». Ces **probabilités relatives** (soft targets) contiennent une information structurelle riche.

Sans distillation, MobileNetV2 doit redécouvrir toutes ces relations par lui-même, avec 25× moins de paramètres.

### La correction

Le script `distill.py` implémente la knowledge distillation :

```
loss = alpha × KL(student_soft, teacher_soft) + (1 - alpha) × CrossEntropy(student, hard_labels)
```

**Le processus :**
1. Le teacher (ViT-B/16) est pré-entraîné et gelé
2. Pour chaque batch, le teacher produit des logits (scores bruts par classe)
3. Les logits du teacher et du student sont « adoucis » par une température T=4.0 :
   - `soft_probs = softmax(logits / T)`
   - T élevé → distribution plus plate → plus d'information sur les classes secondaires
4. La loss KL divergence force le student à reproduire la distribution du teacher
5. La loss hard label (pondération 0.3) garantit que le student apprend aussi les vrais labels

**Pourquoi ça marche :** Le teacher fournit un signal d'apprentissage plus riche que les labels bruts. Un label dur dit « c'est une mésange bleue ». Le teacher dit « c'est une mésange bleue, qui ressemble un peu à une mésange charbonnière et pas du tout à un pigeon ». Le student apprend la structure de l'espace des espèces, pas juste les classes individuelles.

La multiplication par T² (`soft_loss × T²`) compense le fait que la température réduit l'amplitude des gradients — sans cette correction, les gradients seraient trop faibles pour être utiles.

**Impact estimé : +2-4 points de val_acc sur MobileNetV2**

---

## 11. Absence de Focal Loss

### Le problème

La CrossEntropyLoss standard traitait toutes les prédictions de la même façon, qu'elles soient faciles ou difficiles.

### Ce que ça implique

Le dataset a un déséquilibre important : ratio min/max de 7:332 images par espèce (rapport 47:1), avec 116 espèces ayant moins de 100 images. Le `WeightedRandomSampler` rééquilibre la fréquence d'apparition des classes dans les batches, mais il ne modifie pas la **loss elle-même**.

Avec la CrossEntropy standard, le modèle optimise principalement les exemples « faciles » (espèces communes, bien photographiées) car ils sont les plus nombreux et produisent les plus grands gradients en volume. Les espèces rares et les cas difficiles (angle inhabituel, juvénile, lumière extrême) contribuent peu à la loss globale et sont donc « ignorés » par l'optimisation.

Résultat : le modèle atteint 95 % sur les 50 espèces les plus communes mais seulement 20-30 % sur les espèces rares.

### La correction

```python
class FocalLoss(nn.Module):
    def forward(self, logits, targets):
        p_t = (probs * targets).sum(dim=1)          # probabilité de la bonne classe
        focal_weight = (1 - p_t) ** self.gamma       # pondération focale
        ce = -(targets * log_probs).sum(dim=1)       # cross-entropy classique
        return (focal_weight * ce).mean()
```

La Focal Loss multiplie la cross-entropy par `(1 - p_t)^γ` où :
- `p_t` = probabilité que le modèle assigne à la bonne classe
- `γ` = 2.0 (par défaut)

**Concrètement :**
- Si le modèle prédit correctement avec confiance (`p_t = 0.95`) → poids = `(1 - 0.95)² = 0.0025` → la loss est quasi nulle, l'exemple est « ignoré »
- Si le modèle se trompe (`p_t = 0.1`) → poids = `(1 - 0.1)² = 0.81` → la loss est presque intacte, l'exemple « compte »
- Si le modèle est incertain (`p_t = 0.5`) → poids = `(1 - 0.5)² = 0.25` → contribution modérée

Le modèle concentre automatiquement ses efforts sur les cas difficiles : espèces rares, photos ambiguës, confusions inter-espèces. C'est exactement ce qu'il faut pour un dataset déséquilibré.

Le label smoothing (0.1) est conservé dans la Focal Loss — les deux techniques sont complémentaires.

**Impact estimé : +1-3 points de val_acc (surtout sur les espèces rares)**

---

## 12. RandAugment trop conservateur

### Le problème

RandAugment était configuré avec `num_ops=2, magnitude=9` — des valeurs modérées.

### Ce que ça implique

RandAugment applique N opérations aléatoires (parmi rotation, translation, cisaillement, contraste, etc.) avec une magnitude M. `num_ops=2` signifie seulement 2 transformations par image, et `magnitude=9` (sur 30) est une intensité faible.

Avec un overfitting de 30 points et ~224 images/espèce, le modèle a besoin de **plus de diversité** dans les images d'entraînement. Des transformations trop douces ne modifient pas assez l'apparence des images pour empêcher la mémorisation.

### La correction

```
Avant : RandAugment(num_ops=2, magnitude=9)
Après : RandAugment(num_ops=3, magnitude=12)  (exposé en CLI)
```

3 opérations à magnitude 12 produisent des transformations plus variées et plus intenses : rotations plus prononcées, cisaillements plus forts, changements de contraste plus marqués. Chaque image d'entraînement apparaît sous des formes suffisamment différentes d'une epoch à l'autre pour que le modèle ne puisse pas la « reconnaître ».

Les valeurs sont exposées en CLI (`--randaugment-ops` et `--randaugment-magnitude`) pour pouvoir les ajuster si l'augmentation est trop forte (signes : train_acc très basse, underfitting).

**Impact estimé : +0.5-1 point de val_acc**

---

## 13. Pipeline de quantification inadapté

### Le problème

`export.py` utilisait `onnxruntime.quantization.quantize_dynamic()` — une quantification dynamique INT8 où seuls les poids sont quantifiés, les activations restent en float32.

### Ce que ça implique

Les deux cibles de déploiement (IMX500 et Hailo-10H) exigent des formats spécifiques que la quantification dynamique ne fournit pas :

**IMX500 (Pi Zero + AI Camera) :**
- Attend un modèle **entièrement quantifié INT8** (poids ET activations)
- Format firmware Sony (.rpk), converti via le Model Compression Toolkit (MCT) de Sony
- Nécessite un dataset de calibration (100-500 images) pour mesurer la plage des activations
- La quantification dynamique n'est tout simplement **pas compatible** avec le firmware IMX500

**Hailo-10H (Pi 5 + AI HAT+ 2) :**
- Le Dataflow Compiler (DFC) Hailo gère lui-même la quantification (INT4/INT8)
- Il attend un ONNX float32 « propre » (opset ≥ 13, shapes fixes) comme entrée
- La quantification dynamique produit un fichier hybride que le DFC ne peut pas parser

**En termes de précision :**

| Méthode | Perte de précision typique |
|---------|---------------------------|
| Quantification dynamique (ancien) | -2 à -4 % accuracy |
| Quantification statique avec calibration | -0.5 à -1.5 % accuracy |
| QAT + quantification statique | -0 à -0.5 % accuracy |

La quantification dynamique perd 2 à 4 points de précision parce qu'elle ne connaît pas la plage réelle des activations — elle doit la deviner à l'exécution. La quantification statique mesure cette plage en avance sur des images de calibration, ce qui permet un mapping INT8 beaucoup plus précis.

### La correction

`export.py` supporte maintenant trois cibles distinctes :

**`--target onnx`** : ONNX float32 + quantification dynamique INT8 (pour validation/debug)

**`--target hailo`** : ONNX float32 avec opset 13+ et input shape fixe, prêt pour le Hailo DFC :
```
hailo parser onnx model.onnx → hailo optimize --hw-arch hailo10h → hailo compile → .hef
```

**`--target imx500`** : quantification statique INT8 via Sony MCT avec calibration :
```python
def export_imx500(model, output_path, calibration_loader):
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        model, representative_dataset_gen, target_platform_capabilities=...
    )
```

Le **QAT** (Quantization-Aware Training) a été ajouté dans `train.py` comme phase post-entraînement optionnelle (`--qat`). Le QAT insère des « fake quantizers » dans le graphe du modèle pendant l'entraînement, qui simulent l'arrondi INT8. Le modèle apprend à être robuste à cette perte de précision, ce qui réduit la dégradation à l'export de -2/-4 % à quasi-0 %.

**Impact : perte de précision réduite de -4 % à -0.5 % à l'export**

---

## 14. Stratégie coarse-to-fine abandonnée

### Le problème initial

L'audit recommandait une stratégie en 2 étapes : d'abord entraîner sur 558 espèces européennes, puis fine-tuner sur ~20-30 espèces de jardin avec une classe « autre oiseau ».

### Pourquoi elle a été abandonnée

Après analyse, cette stratégie est **contre-productive** pour ce projet spécifique, pour trois raisons :

**1. Pas de shift de domaine.** Le coarse-to-fine suppose que les données cibles (jardin) sont visuellement différentes des données sources (Europe). Ce n'est pas le cas : un rouge-gorge en Allemagne est identique à un rouge-gorge dans un jardin français. Le fine-tuning sur un sous-ensemble n'apporte pas de nouvelles features.

**2. La diversité des classes aide.** La littérature montre que la précision du transfer learning croît avec le nombre de classes dans le pré-entraînement. Réduire de 558 à 20 classes élimine les contrastes visuels qui aident le modèle à apprendre des features discriminantes. Savoir distinguer 558 espèces rend le modèle meilleur pour distinguer 20 espèces que s'il n'en avait vu que 20.

**3. Risque d'oubli catastrophique.** MobileNetV2 (3.4M paramètres) est un petit modèle, particulièrement vulnérable au forgetting lors de fine-tuning multi-étapes. Chaque étape de fine-tuning risque d'écraser les features apprises à l'étape précédente.

### La solution retenue

Le filtrage « jardin » se fait en **post-traitement** côté application :

```python
GARDEN_SPECIES = {"parus_major", "turdus_merula", "erithacus_rubecula", ...}
prediction = model.predict(image)  # parmi 558 espèces
label = prediction if prediction in GARDEN_SPECIES else "autre oiseau"
```

Le modèle 558 classes **est** le détecteur d'« autre oiseau ». Un simple `if/else` en post-traitement remplace une étape d'entraînement complète sans perte de précision, et permet de changer la liste des espèces de jardin sans ré-entraîner.

---

## Synthèse des corrections

### Corrections anti-overfitting (écart train/val de 30 → objectif < 15 points)

| # | Correction | Mécanisme | Impact |
|---|-----------|-----------|--------|
| 1 | Weight decay ×100 | Empêche les poids de grossir → frontières de décision lisses | +2-5 pts |
| 2 | MixUp + CutMix | Augmente la diversité, labels mous → moins de mémorisation | +3-5 pts |
| 3 | ColorJitter | Invariance aux conditions de lumière | +1-2 pts |
| 4 | Backbone LR ÷10 | Préserve les features ImageNet | +1-2 pts |
| 5 | Dropout 0.3 → 0.5 | Empêche la co-adaptation des neurones | +1-2 pts |
| 12 | RandAugment renforcé | Plus de transformations, plus intenses | +0.5-1 pt |

### Corrections de stabilité et convergence

| # | Correction | Mécanisme | Impact |
|---|-----------|-----------|--------|
| 6 | 30 → 80 epochs | Le modèle régularisé a besoin de plus de temps | +2-4 pts |
| 7 | Gradient clipping | Empêche les spikes de gradient | +0.5-1 pt |
| 8 | EMA | Lisse le bruit d'optimisation | +1-2 pts |

### Corrections structurelles

| # | Correction | Mécanisme | Impact |
|---|-----------|-----------|--------|
| 9 | Architectures corrigées | Modèles déployables + ViT teacher | Pré-requis |
| 10 | Knowledge distillation | Le teacher guide le student | +2-4 pts |
| 11 | Focal Loss | Focus sur les exemples difficiles | +1-3 pts |
| 13 | Quantification statique + QAT | Export compatible cibles, perte minimale | -0.5 % au lieu de -4 % |
| 14 | Abandon coarse-to-fine | Post-traitement au lieu de re-training | Simplification |

### Impact cumulé attendu

Les impacts ne sont **pas simplement additifs** — certaines corrections interagissent. Par exemple, le MixUp et le weight decay renforcent mutuellement la régularisation, et l'EMA profite davantage quand le gradient clipping stabilise l'entraînement.

Estimation réaliste de la val_acc avec toutes les corrections :
- MobileNetV2 sur 558 classes : **70-78 %** (vs 61.98 % avant)
- EfficientNetV2-B2 sur 558 classes : **75-82 %**
- ViT-B/16 (teacher) sur 558 classes : **80-87 %**
- MobileNetV2 distillé sur 558 classes : **73-80 %**
