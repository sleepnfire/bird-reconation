# Audit : Recommandations d'entraînement vs Implémentation actuelle

Date : 2026-06-13

## Objectif

Comparer les bonnes pratiques reconnues pour l'entraînement de modèles de classification fine-grained (reconnaissance d'espèces d'oiseaux) avec l'implémentation actuelle de `train.py`, en identifiant les écarts et leur impact sur les performances.

---

## Résultats actuels — Diagnostic

| Métrique | Valeur |
|----------|--------|
| Architecture | MobileNetV2 (3.5M paramètres) |
| Dataset | 558 espèces européennes, ~124 911 images (~224 images/espèce) |
| train_acc | 92.07% |
| val_acc | **61.98%** |
| Écart train/val | **30.09 points** |
| train_loss | 0.2874 |
| val_loss | 1.8607 |
| Epochs | 30 (best = epoch 30, modèle encore en amélioration) |

**Constat** : overfitting sévère. Le modèle mémorise les exemples d'entraînement sans généraliser. La val_loss stagne autour de 1.86-1.93 dès l'epoch 9-10 alors que la train_loss continue de chuter. L'early stopping (patience=7) ne s'est jamais déclenché car la val_acc continuait de monter lentement grâce à une confiance accrue sur les exemples faciles.

---

## 1. Régularisation — Weight Decay

### Ce qui est fait

`weight_decay=1e-4` codé en dur (`train.py:491`), non exposé en CLI.

### Ce qui est recommandé

Pour le fine-tuning de modèles pré-entraînés sur des tâches fine-grained avec peu de données par classe (~224 images), le weight decay optimal se situe entre **1e-2 et 5e-2** pour les CNN, et **0.05 à 0.3** pour les Vision Transformers.

La valeur actuelle de 1e-4 est **100x trop faible** — elle n'apporte quasiment aucune régularisation.

### Sources

- Li et al., *Rethinking the Hyperparameters for Fine-tuning*, ICLR 2020 — montre que les hyperparamètres par défaut (dont weight_decay=1e-4) ne sont pas optimaux pour le fine-tuning, et que le weight decay optimal dépend de la similarité source/cible. [arXiv:2002.11770](https://arxiv.org/abs/2002.11770)
- Han et al., *Weight Decay Improves Language Model Plasticity*, 2026 — confirme que le weight decay améliore la plasticité et les performances downstream. [arXiv:2602.11137](https://arxiv.org/abs/2602.11137)
- Discussion PyTorch Forums — confirme que weight_decay=1e-2 est standard pour le fine-tuning. [PyTorch Forums](https://discuss.pytorch.org/t/about-learning-rate-and-weight-decay-fine-tuning/130849)

### Impact estimé : +2-5 points val_acc

---

## 2. MixUp et CutMix

### Ce qui est fait

Aucune technique de mélange inter-échantillons. L'augmentation opère uniquement au niveau de chaque image individuelle.

### Ce qui est recommandé

**MixUp** (alpha=0.2) interpole linéairement deux images et leurs labels. **CutMix** (alpha=1.0) découpe une région d'une image et la colle sur une autre, en mélangeant les labels proportionnellement à la surface.

Sur CUB-200-2011 (dataset de référence pour la classification d'oiseaux, 200 espèces), les résultats publiés montrent :

| Méthode | ResNeXt-50 Top-1 sur CUB-200 |
|---------|-------------------------------|
| Baseline | ~82% |
| MixUp | 84.58% |
| CutMix | 85.68% |
| AutoMix | +2.19-3.55% vs baseline |

Ces techniques s'appliquent **au niveau batch** dans la boucle d'entraînement (pas dans `get_transforms()`), et nécessitent des soft labels (vecteurs de probabilité) au lieu de hard labels (indices).

### Sources

- Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018 — papier fondateur, montre l'amélioration de généralisation sur ImageNet, CIFAR-10/100. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*, ICCV 2019 — +1.98% sur CIFAR-100, surpasse MixUp et Cutout. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
- OpenMixup Benchmarks — benchmarks fine-grained (CUB-200, FGVC-Aircraft) comparant MixUp, CutMix, AutoMix et variantes. [OpenMixup](https://openmixup.readthedocs.io/en/latest/mixup_benchmarks/Mixup_downstream.html)

### Impact estimé : +3-5 points val_acc

---

## 3. Data Augmentation — ColorJitter

### Ce qui est fait

```python
RandomResizedCrop(image_size, scale=(0.6, 1.0))
RandomHorizontalFlip()
RandAugment(num_ops=2, magnitude=9)
RandomErasing(p=0.25)
```

### Ce qui manque

**ColorJitter** est critique pour les oiseaux car le plumage (couleur, saturation, luminosité) est le critère discriminant principal. Les conditions d'éclairage varient fortement (ombre, soleil direct, couvert, aube/crépuscule). Sans ColorJitter, le modèle risque d'apprendre des biais liés à l'éclairage plutôt que les patterns de couleur intrinsèques au plumage.

Valeurs recommandées : `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)` — le hue est conservatif (0.05) car trop de variation de teinte rendrait les espèces indistinguables.

De plus, `RandAugment(num_ops=2, magnitude=9)` est modéré. Pour les tâches fine-grained avec overfitting sévère, **magnitude 12-15** et **3 ops** sont plus standards.

### Sources

- Cubuk et al., *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space*, NeurIPS 2020 — recommande d'augmenter la magnitude avec la taille du dataset et la capacité du modèle
- Transfert learning guides — ColorJitter est systématiquement listé dans les pipelines d'augmentation pour la classification fine-grained d'espèces naturelles

### Impact estimé : +1-2 points val_acc

---

## 4. Learning Rate — Backbone trop agressif

### Ce qui est fait

- Head LR : 1e-3
- Backbone LR : 1e-4 (`backbone_lr_factor=0.1`)

### Ce qui est recommandé

Avec seulement ~224 images/classe, un backbone LR de 1e-4 modifie trop agressivement les features pré-entraînées ImageNet. Le modèle « oublie » les représentations générales pour mémoriser les exemples spécifiques.

Recommandation : **`backbone_lr_factor=0.01`** (backbone LR = 1e-5). Pour ViT-B/16, un backbone LR encore plus faible (2e-6 à 5e-6) est nécessaire.

### Sources

- Li et al., *Rethinking the Hyperparameters for Fine-tuning*, ICLR 2020 — le LR optimal pour le fine-tuning dépend de la similarité source/cible, et des LR plus faibles sont nécessaires quand le dataset cible est petit. [arXiv:2002.11770](https://arxiv.org/abs/2002.11770)
- Swain, *Fine-Tuning Pre-Trained Models the Right Way*, 2026 — guide pratique sur le differential LR, recommande un facteur 0.01-0.1 pour le backbone. [Medium](https://lalatenduswain.medium.com/fine-tuning-pre-trained-models-the-right-way-a-step-by-step-guide-to-learning-rate-strategy-b3d9c0307222)
- Fine-tuning ViT recommandations — lr=5e-5 avec weight_decay=0.01 pour AdamW, batch size 64. [Medium](https://medium.com/@supersjgk/fine-tuning-vision-transformer-with-hugging-face-and-pytorch-df19839d5396)

### Impact estimé : +1-2 points val_acc

---

## 5. Dropout

### Ce qui est fait

`dropout=0.3` sur la tête de classification uniquement.

### Ce qui est recommandé

Avec 558 classes et un overfitting de 30 points, **dropout=0.4-0.5** est plus approprié. Pour les Vision Transformers, un dropout d'attention de 0.1 est également recommandé.

### Sources

- Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR 2014 — papier fondateur, recommande 0.5 comme valeur par défaut pour les couches fully-connected
- Pratique standard en fine-grained classification — les implémentations de référence utilisent 0.4-0.5 sur le classifier head

### Impact estimé : +1-2 points val_acc

---

## 6. Nombre d'epochs

### Ce qui est fait

30 epochs, early stopping patience=7. Le modèle était encore en amélioration à l'epoch 30.

### Ce qui est recommandé

Avec une régularisation renforcée (weight decay 1e-2, MixUp/CutMix, dropout 0.5), le modèle apprend plus lentement mais généralise mieux. **80-100 epochs** avec **patience=15** sont nécessaires pour atteindre la convergence.

Le cosine annealing sur 100 epochs donne une phase d'exploration plus longue à LR élevé, suivie d'une convergence fine.

### Impact estimé : +2-4 points val_acc (en combinaison avec la régularisation renforcée)

---

## 7. Gradient Clipping

### Ce qui est fait

Aucun gradient clipping dans `train_one_epoch()` (`train.py:283`).

### Ce qui est recommandé

Avec 558 classes et des classes rares (certaines avec seulement 7 images), des spikes de gradient peuvent corrompre les features apprises. `clip_grad_norm_(model.parameters(), max_norm=1.0)` est le standard pour les Transformers et fonctionne bien avec les CNN aussi.

Le clipping doit être appliqué **après** `scaler.unscale_(optimizer)` et **avant** `scaler.step(optimizer)` en mode AMP.

### Sources

- Implémentation BERT (Devlin et al., 2019) — le code officiel utilise `clip_grad_norm_(max_norm=1.0)`, devenu standard pour les Transformers
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — architecture Transformer de référence
- GeeksforGeeks, *Gradient Clipping in PyTorch: Methods, Implementation, and Best Practices* — recommande de monitorer les normes de gradient et d'utiliser le 90e-95e percentile comme seuil. [GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/gradient-clipping-in-pytorch-methods-implementation-and-best-practices/)
- Shadecoder, *Gradient Norm Clipping: A Comprehensive Guide for 2025* — 1.0 est le défaut standard pour les modèles Transformer et CNN fine-tuning. [Shadecoder](https://www.shadecoder.com/topics/gradient-norm-clipping-a-comprehensive-guide-for-2025)

### Impact estimé : +0.5-1 point val_acc (stabilité d'entraînement)

---

## 8. Exponential Moving Average (EMA)

### Ce qui est fait

Non implémenté.

### Ce qui est recommandé

L'EMA maintient une copie lissée des poids du modèle (decay=0.9999) qui moyenne le bruit d'entraînement. Les poids EMA sont utilisés pour la validation et l'export final. L'EMA améliore la généralisation, la robustesse aux labels bruités, la calibration et le transfer learning.

### Sources

- Morales-Brotons et al., *Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits*, 2024 — étude systématique montrant que l'EMA améliore toujours la généralisation par rapport au SGD baseline, avec une convergence plus rapide. [arXiv:2411.18704](https://arxiv.org/abs/2411.18704)
- Pratique standard dans les pipelines d'entraînement modernes (timm, torchvision recipes)

### Impact estimé : +1-2 points val_acc

---

## 9. Architectures — Écart avec les recommandations

### Ce qui est fait

Seuls **MobileNetV2** (72.0% ImageNet) et **EfficientNet-B0** (77.1% ImageNet) sont implémentés.

### Ce qui est recommandé (cf. `deployment-model-zoo.md`)

| Modèle | ImageNet Top-1 | Cible | Statut | Disponibilité torchvision |
|--------|---------------|-------|--------|---------------------------|
| MobileNetV2 | 72.0% | IMX500 + Hailo | ✅ | `models.mobilenet_v2` |
| EfficientNet-B0 | 77.1% | IMX500 | ✅ | `models.efficientnet_b0` |
| **EfficientNetV2-B2** | **~80%** | **IMX500** | ❌ | `models.efficientnet_v2_s` |
| **ViT-B/16** | **84.5%** | **Hailo-10H** | ❌ | `models.vit_b_16` |

**EfficientNetV2-B2** utilise des blocs Fused-MBConv en début de réseau (meilleure capture des détails spatiaux) et le progressive learning. L'écart de 3 points sur ImageNet se traduit typiquement par **4-6 points** sur les tâches fine-grained. Le modèle est déjà pré-converti en `.rpk` dans le Model Zoo IMX500.

**ViT-B/16** excelle en classification fine-grained grâce au mécanisme d'attention qui se focalise naturellement sur les parties discriminantes. Cependant, il nécessite des hyperparamètres différents : weight_decay=0.05-0.3, LR plus faible, et une séparation backbone/head différente (`model.encoder` / `model.heads`).

### Sources

- Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021
- Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021
- Roboflow, *EfficientNet vs. MobileNet V2 Classification: Compared and Contrasted* — EfficientNetV2 surpasse systématiquement MobileNetV2 en précision. [Roboflow](https://roboflow.com/compare/efficientnet-vs-mobilenet-v2-classification)
- Étude comparative edge : *Comparative and edge-hybrid modeling of EfficientNetV2 and MobileNetV2*, Journal of Edge Computing, 2025. [ResearchGate](https://www.researchgate.net/publication/396731090)
- Model Zoo IMX500 : [github.com/raspberrypi/imx500-models](https://github.com/raspberrypi/imx500-models)
- Model Zoo Hailo-10H : [github.com/hailo-ai/hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo)

---

## 10. Stratégie coarse-to-fine (2 étapes)

### Ce qui est fait

Seule l'étape 1 (fine-tuning sur 558 espèces européennes) est implémentée. L'étape 2 (spécialisation sur 20-30 espèces de jardin + classe "autre oiseau") n'existe pas dans le code.

### Ce qui est recommandé

La stratégie en 2 étapes est validée par la littérature. Sur CUB-200-2011, l'approche hiérarchique (sous-ensembles de classes visuellement similaires avec classifieurs locaux) améliore la précision de 64.5% à 72.7% (+12.7% relatif).

Pour l'étape 2, les points clés sont :

1. **Construction de la classe "autre oiseau"** : ne pas prendre des espèces aléatoires, mais les espèces **confondantes** identifiées via la matrice de confusion de l'étape 1 + espèces géographiquement plausibles en France
2. **LR très faible** : tête 1e-4, backbone 1e-6 — l'objectif est d'affiner les frontières de décision, pas de réapprendre les features
3. **Gel plus profond du backbone** : geler 14/19 blocs conv pour MobileNetV2
4. **Remplacer la tête** : 558 → ~35 classes (20-30 jardin + 1 "autre oiseau")
5. **Ne pas simplement fine-tuner** le checkpoint 558 classes → remplacer la tête et recommencer le head training

### Sources

- Ge et al., *Fine-grained bird species recognition via hierarchical subset learning*, ICIP 2015 — amélioration de 64.5% à 72.7% sur CUB-200-2011 avec classification hiérarchique. [IEEE Xplore](https://ieeexplore.ieee.org/document/7350861/)
- Two-Stage Fine-Tuning Strategy — ressource émergente consolidant les pratiques de fine-tuning en 2 étapes. [Emergent Mind](https://www.emergentmind.com/topics/two-stage-fine-tuning-strategy)
- Coarse2Fine — système avec coarse-stage attention + fine-grained classifier, 89.5% sur CUB-200-2011

---

## 11. Knowledge Distillation

### Ce qui est fait

Non implémenté.

### Ce qui est recommandé

Une fois le ViT-B/16 entraîné (pour le Hailo-10H), utiliser ses prédictions comme « soft targets » pour améliorer le MobileNetV2 (pour l'IMX500). La distillation ajoute une loss de KL-divergence entre les logits du teacher (ViT) et du student (MobileNetV2), avec une température T=4.0.

Pondération typique : `alpha=0.7 * distillation_loss + 0.3 * hard_label_loss`.

Un papier de 2023 (Beijing Forestry University) a spécifiquement appliqué la distillation découplée à la classification fine-grained d'oiseaux, créant un modèle léger performant pour les appareils edge.

### Sources

- Wang et al., *A Fine-Grained Bird Classification Method Based on Attention and Decoupled Knowledge Distillation*, Animals 2023 — distillation appliquée à la classification d'oiseaux pour appareils edge. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9854642/)
- Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015 — papier fondateur de la knowledge distillation

### Impact estimé : +2-4 points val_acc sur le modèle student (MobileNetV2)

---

## 12. Focal Loss / Class-Balanced Loss

### Ce qui est fait

`CrossEntropyLoss(label_smoothing=0.1)` + `WeightedRandomSampler` pour le rééquilibrage.

### Ce qui est recommandé

Le `WeightedRandomSampler` rééquilibre la fréquence d'échantillonnage mais ne modifie pas la loss. Avec 116 espèces ayant moins de 100 images et un ratio min/max de 7:332 (47:1), la **Focal Loss** est recommandée en complément. Elle réduit le poids des exemples faciles et concentre l'apprentissage sur les exemples difficiles (souvent les classes rares ou les confusions inter-espèces).

Le label smoothing (0.1) est bon et doit être conservé — il empêche les prédictions trop confiantes et améliore la généralisation.

### Sources

- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017 — papier fondateur, introduit le facteur modulant (1-p_t)^gamma pour down-weighter les exemples faciles
- Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019 — propose un rééquilibrage basé sur le nombre effectif d'échantillons, spécifiquement conçu pour les distributions long-tail. [arXiv:1901.05555](https://arxiv.org/pdf/1901.05555)
- iNaturalist 2018 — dataset de référence pour la classification long-tail (8142 espèces, ratio d'imbalance 500:1), utilise systématiquement des stratégies de rééquilibrage de loss
- Label smoothing : Müller et al., *When Does Label Smoothing Help?*, NeurIPS 2019 ; *Delving Deep Into Label Smoothing*, IEEE TIP 2021. [IEEE](https://dl.acm.org/doi/abs/10.1109/TIP.2021.3089942)

### Impact estimé : +1-3 points val_acc (surtout sur les classes rares)

---

## 13. Pipeline de quantification

### Ce qui est fait

`export.py` utilise `onnxruntime.quantization.quantize_dynamic()` (quantification dynamique INT8) — seuls les poids sont quantifiés, les activations restent en float32.

### Ce qui est recommandé

La quantification dynamique est **inadaptée** pour les deux cibles de déploiement :

**IMX500** : Le firmware attend un modèle **entièrement quantifié INT8** (poids + activations). Le pipeline correct est :

```
PyTorch (.pth) → ONNX float32 → Sony MCT (Model Compression Toolkit)
  → Quantification statique INT8 avec données de calibration
  → IMX500 Converter → Firmware .rpk
```

Le MCT (Model Compression Toolkit) de Sony utilise la quantification statique par calibration : quantification symétrique per-channel pour les poids, per-tensor pour les activations. Un dataset de calibration de 100-500 images représentatives est nécessaire.

**Hailo-10H** : Le Hailo Dataflow Compiler gère la quantification (INT4/INT8) et la compilation en .hef :

```
PyTorch (.pth) → ONNX float32 → Hailo DFC
  → Parse → Optimize (quantification INT4/INT8) → Compile → .hef
```

**QAT (Quantization-Aware Training)** est fortement recommandé pour l'IMX500 car le budget de 8 Mo SRAM est serré. Le QAT simule le bruit de quantification pendant l'entraînement, permettant au modèle de s'adapter à la perte de précision.

| Méthode | Perte de précision typique |
|---------|---------------------------|
| Quantification dynamique (actuel) | -2 à -4% accuracy |
| Quantification statique avec calibration | -0.5 à -1.5% accuracy |
| QAT + quantification statique | -0 à -0.5% accuracy |

### Sources

- Weights & Biases, *Quantization-Aware Training: Empowering efficient AI on edge devices* — comparaison QAT vs PTQ, montre que le PTQ peut faire chuter ResNet-50 de 75% à 50%, tandis que le QAT maintient la précision proche du float. [W&B](https://wandb.ai/onlineinference/qat/reports/Quantization-Aware-Training-Empowering-efficient-AI-on-edge-devices--VmlldzoxMTcyOTEwMA)
- SabrePC, *What is Quantization Aware Training? QAT vs. PTQ* — guide pratique comparant les deux approches pour le déploiement edge. [SabrePC](https://www.sabrepc.com/blog/deep-learning-and-ai/what-is-quantization-aware-training-qat-vs-ptq)
- Sony MCT (Model Compression Toolkit) : [github.com/sony/model_optimization](https://github.com/sony/model_optimization/releases)
- Raspberry Pi AI Camera model conversion : [github.com/raspberrypi/documentation](https://github.com/raspberrypi/documentation/blob/develop/documentation/asciidoc/accessories/ai-camera/model-conversion.adoc)
- Hailo DFC User Guide v3.27.0 : [Documentation PDF](https://mmmsk.ai.kr/Projects/Embedded-AI/files/hailo_dataflow_compiler_v3.27.0_user_guide.pdf)
- Hailo Model Zoo : [github.com/hailo-ai/hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/GETTING_STARTED.rst)

---

## ERRATUM — Mapping EfficientNetV2-B2 / EfficientNetV2-S (corrigé le 2026-06-13)

### Erreur initiale

L'item 9 de cet audit recommandait **EfficientNetV2-B2** comme modèle IMX500, mappé vers `models.efficientnet_v2_s` de torchvision. Ce mapping est **incorrect** :

| Modèle | Params | FLOPs | Top-1 ImageNet | Taille INT8 | Disponibilité |
|--------|--------|-------|----------------|-------------|---------------|
| EfficientNetV2-**B2** | **10.1M** | 1.7G | 80.5% | **~6.5 Mo** (RPK) | **timm** (`tf_efficientnetv2_b2.in1k`) |
| EfficientNetV2-**S** | **21.5M** | 8.4G | 83.9% | **~21.5 Mo** | **torchvision** (`models.efficientnet_v2_s`) |

EfficientNetV2-S est **2.1x plus gros** que le B2. Avec ~21.5 Mo en INT8, il **ne rentre pas** dans les 8 Mo de SRAM de l'IMX500. Il n'est présent **dans aucun** des deux Model Zoos (ni IMX500, ni Hailo-10H). Le code implémenté avec `efficientnet_v2_s` n'est déployable nulle part.

### Correction

Les variantes EfficientNetV2-B0/B1/B2 ne sont **pas dans torchvision** (qui ne fournit que S/M/L). Elles sont disponibles dans **timm** :

```python
import timm
model = timm.create_model('tf_efficientnetv2_b2.in1k', pretrained=True, num_classes=558)
```

Pour utiliser EfficientNetV2-B2 dans `train.py`, il faut ajouter `timm` comme dépendance.

### Sources

- Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021 — définit les variantes B0-B3 et S/M/L avec des paramètres différents. [arXiv:2104.00298](https://arxiv.org/abs/2104.00298)
- torchvision efficientnet.py — ne contient que `efficientnet_v2_s`, `_m`, `_l`. [GitHub](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)
- timm model registry — contient `tf_efficientnetv2_b0/b1/b2/b3.in1k`. [HuggingFace](https://huggingface.co/timm/tf_efficientnetv2_b2.in1k)
- IMX500 Model Zoo — `efficientnetv2_b2.rpk` = **6.51 Mo**, top-1 quantifié = **77.7%**. [GitHub](https://github.com/raspberrypi/imx500-models)

---

## 14. Alignement modèles ↔ plateformes de déploiement

### Constat

L'entraînement d'un modèle ne sert que s'il est déployable sur la plateforme cible. Chaque architecture dans `train.py` doit correspondre à un modèle validé dans le Model Zoo du constructeur.

### État actuel après implémentation

| Modèle dans train.py | IMX500 (8 Mo SRAM) | Hailo-10H (40 TOPS) | Verdict |
|---|:---:|:---:|---|
| `mobilenetv2` | ✅ 3.89 Mo RPK, 71.6% | ✅ 71.0% | Baseline cross-platform |
| `efficientnet_b0` | ✅ 5.99 Mo RPK, 72.1% | ❌ absent | IMX500 uniquement |
| `efficientnet_v2_s` | ❌ **trop gros** (21.5M params) | ❌ **absent** | **À remplacer** par V2-B2 (timm) |
| `vit_b_16` | ❌ trop gros | ✅ 83.6%, meilleur du Model Zoo | Hailo uniquement |

### Modèles recommandés par plateforme

**IMX500 (Pi Zero + AI Camera)** — Top 5 du Model Zoo par précision INT8 :

| Modèle | Top-1 (quantifié) | Taille RPK | Entraînable | Résolution |
|--------|-------------------|------------|:-----------:|------------|
| **EfficientNetV2-B2** | **77.7%** | 6.51 Mo | timm | 260×260 |
| EfficientNetV2-B1 | 77.0% | 6.37 Mo | timm | 240×240 |
| EfficientNetV2-B0 | 76.7% | 6.52 Mo | timm | 224×224 |
| EfficientNet Lite-0 | 75.3% | 5.32 Mo | timm | 224×224 |
| RegNetY-004 | 73.8% | — | torchvision | 224×224 |

**Hailo-10H (Pi 5 + AI HAT+ 2)** — Top 5 du Model Zoo entraînables :

| Modèle | Top-1 (quantifié) | Params | Entraînable | Résolution |
|--------|-------------------|--------|:-----------:|------------|
| **ViT-B/16** | **83.6%** | 86.6M | torchvision | 224×224 |
| ViT-Large | 82.5% | 304M | torchvision | 224×224 |
| Swin-Small | 80.0% | 50M | torchvision | 224×224 |
| Swin-Tiny | 79.4% | 28M | torchvision | 224×224 |
| ResNeXt50_32x4d | 78.4% | 25M | torchvision | 224×224 |

### Hyperparamètres recommandés par architecture

Les architectures CNN et Transformer nécessitent des hyperparamètres différents pour le fine-tuning :

| Hyperparamètre | CNN (MobileNetV2, EfficientNet) | ViT-B/16 |
|---|---|---|
| Optimiseur | AdamW | AdamW |
| Head LR | 1e-3 | 1e-4 |
| Backbone LR factor | 0.01 | 0.01 |
| Weight decay | **1e-2** | **0.05** |
| Dropout (head) | 0.5 | 0.5 |
| Warmup epochs | 3 | 5–10 |
| MixUp alpha | 0.2 | 0.8 |
| CutMix alpha | 1.0 | 1.0 |
| Label smoothing | 0.1 | 0.1 |
| Gradient clipping | 1.0 | 1.0 |
| Epochs | 80 | 50–80 |

Le ViT nécessite un weight decay plus élevé (0.05 vs 1e-2) et un LR plus faible car les Transformers sont plus sensibles aux hyperparamètres que les CNN.

### Sources

- Hailo Model Zoo — classification Hailo-10H, ViT-Base à 84.5% float (83.6% quantifié). [GitHub](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_classification.rst)
- IMX500 Model Zoo — 15 modèles pré-convertis .rpk. [GitHub](https://github.com/raspberrypi/imx500-models)
- Steiner et al., *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers*, 2021 — weight_decay=0.1, warmup=10k steps pour le pré-entraînement ViT. [arXiv:2106.10270](https://arxiv.org/abs/2106.10270)
- Touvron et al., *Training data-efficient image transformers & distillation through attention* (DeiT), ICML 2021 — recommande AdamW, wd=0.05, lr=5e-4, MixUp 0.8, CutMix 1.0. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)

### Action requise

Remplacer `efficientnet_v2_s` dans `train.py` par `efficientnetv2_b2` via timm (ajouter `timm` aux dépendances). Alternativement, conserver uniquement `efficientnet_b0` + `mobilenetv2` pour l'IMX500 sans nouvelle dépendance.

---

## 15. Stratégie d'entraînement — pas de spécialisation jardin

### Décision

L'entraînement se fait **uniquement** sur le dataset Europe (558 espèces). L'étape 2 de spécialisation "jardin" (20-30 espèces) n'est **pas implémentée**. Le filtrage des espèces de jardin se fait en **post-traitement** côté application :

```python
GARDEN_SPECIES = {"parus_major", "turdus_merula", "erithacus_rubecula", ...}
prediction = model.predict(image)  # une des 558 espèces
label = prediction if prediction in GARDEN_SPECIES else "autre oiseau"
```

### Justification

1. **Pas de shift de domaine** entre Europe/France/Jardin — un rouge-gorge photographié en France est visuellement identique à un rouge-gorge allemand. Le "gradual fine-tuning" n'a de sens que quand les distributions de domaine sont réellement différentes.

2. **La diversité du dataset aide** — la précision du transfer learning croît avec le nombre de classes sources. Réduire de 558 à ~300 (France) puis ~20 (jardin) dégrade les features contrastives.

3. **Risque d'oubli catastrophique** — les modèles < 100M paramètres (MobileNetV2 = 3.5M) sont particulièrement vulnérables au forgetting lors de fine-tuning multi-étapes.

4. **Le modèle Europe EST le détecteur d'« autre oiseau »** — il identifie déjà les 558 espèces. Un simple `if/else` en post-traitement remplace une étape d'entraînement complète sans perte de précision.

### Sources

- Cui et al., *Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning*, CVPR 2018 — la similarité de domaine source/cible détermine la qualité du transfert, pas la granularité géographique. [arXiv:1806.06193](https://arxiv.org/abs/1806.06193)
- *On Transfer in Classification: How Well do Subsets of Classes Generalize?*, 2024 — la précision croît avec le nombre de classes dans le sous-ensemble de pré-entraînement. [arXiv:2403.03569](https://arxiv.org/abs/2403.03569)
- *Efficient Rehearsal for Catastrophic Forgetting in Multi-stage Fine-tuning*, 2024 — les petits modèles sont plus vulnérables à l'oubli catastrophique. [arXiv:2402.08096](https://arxiv.org/pdf/2402.08096)

---

## Synthèse — Tableau récapitulatif (mis à jour 2026-06-13)

| # | Aspect | Actuel | Recommandé | Impact estimé | Effort | Statut |
|---|--------|--------|-----------|---------------|--------|--------|
| 1 | Weight decay | 1e-4 | **1e-2** | +2-5 pts | Faible | ✅ Implémenté |
| 2 | MixUp/CutMix | Absent | **MixUp(0.2) + CutMix(1.0)** | +3-5 pts | Moyen | ✅ Implémenté |
| 3 | ColorJitter | Absent | **brightness/contrast/sat=0.3, hue=0.05** | +1-2 pts | Faible | ✅ Implémenté |
| 4 | Backbone LR factor | 0.1 | **0.01** | +1-2 pts | Faible | ✅ Implémenté |
| 5 | Dropout | 0.3 | **0.5** | +1-2 pts | Faible | ✅ Implémenté |
| 6 | Epochs | 30 | **80** | +2-4 pts | Faible | ✅ Implémenté |
| 7 | Gradient clipping | Absent | **max_norm=1.0** | +0.5-1 pt | Faible | ✅ Implémenté |
| 8 | EMA | Absent | **decay=0.9999** | +1-2 pts | Moyen | ✅ Implémenté |
| 9 | Architecture IMX500 | EfficientNet-B0 | **EfficientNetV2-B2** (timm) | +5-6 pts vs B0 | Moyen | ✅ Implémenté — `efficientnetv2_b2` via timm (`tf_efficientnetv2_b2.in1k`) |
| 10 | Architecture Hailo | Absent | **ViT-B/16** | +12 pts vs MobileNetV2 | Moyen | ✅ Implémenté |
| 11 | Étape 2 (garden) | Absent | ~~À implémenter~~ | — | — | ❌ **Annulé** — filtrage post-traitement |
| 12 | Knowledge distillation | Absent | **ViT → MobileNetV2** | +2-4 pts sur student | Élevé | ✅ Implémenté — `distill.py` (script séparé, 28 tests) |
| 13 | Focal/CB-Loss | Absent | **À ajouter** | +1-3 pts (classes rares) | Moyen | ✅ Implémenté |
| 14 | Quantification | Dynamique | **Statique + QAT** | -0.5% vs -4% accuracy | Élevé | 🔜 À faire |
| 15 | RandAugment | (2, 9) | **(3, 12)** | +0.5-1 pt | Faible | ✅ Implémenté |

### Priorisation révisée

**✅ Terminé — Anti-overfitting + architectures (items 1-10, 12, 13, 15)** : toutes les régularisations, les 4 architectures (MobileNetV2, EfficientNet-B0, EfficientNetV2-B2, ViT-B/16) et la knowledge distillation sont implémentés.

**🔜 À faire — Quantification (item 14)** : refondre `export.py` pour utiliser Sony MCT (IMX500) et Hailo DFC (Hailo-10H), ajouter le QAT.

---

## Références complètes

### Papiers fondateurs

1. Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
2. Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*, ICCV 2019. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
3. Li et al., *Rethinking the Hyperparameters for Fine-tuning*, ICLR 2020. [arXiv:2002.11770](https://arxiv.org/abs/2002.11770)
4. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
5. Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019. [arXiv:1901.05555](https://arxiv.org/pdf/1901.05555)
6. Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015
7. Müller et al., *When Does Label Smoothing Help?*, NeurIPS 2019
8. *Delving Deep Into Label Smoothing*, IEEE TIP 2021. [IEEE](https://dl.acm.org/doi/abs/10.1109/TIP.2021.3089942)
9. Morales-Brotons et al., *Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits*, 2024. [arXiv:2411.18704](https://arxiv.org/abs/2411.18704)
10. Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR 2014

### Classification fine-grained d'oiseaux

11. Wang et al., *A Fine-Grained Bird Classification Method Based on Attention and Decoupled Knowledge Distillation*, Animals 2023. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9854642/)
12. Ge et al., *Fine-grained bird species recognition via hierarchical subset learning*, ICIP 2015. [IEEE Xplore](https://ieeexplore.ieee.org/document/7350861/)
13. Mochurad et al., *A New Efficient Classifier for Bird Classification Based on Transfer Learning*, J. Engineering 2024. [Wiley](https://onlinelibrary.wiley.com/doi/10.1155/2024/8254130)
14. *Bird Species Detection Net*, PMC 2025. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11723449/)

### Architectures

15. Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021
16. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021

### Quantification et déploiement edge

17. Sony MCT (Model Compression Toolkit). [GitHub](https://github.com/sony/model_optimization/releases)
18. Raspberry Pi AI Camera — Model Conversion. [Documentation](https://github.com/raspberrypi/documentation/blob/develop/documentation/asciidoc/accessories/ai-camera/model-conversion.adoc)
19. Hailo Model Zoo. [GitHub](https://github.com/hailo-ai/hailo_model_zoo)
20. Hailo DFC User Guide v3.27.0. [PDF](https://mmmsk.ai.kr/Projects/Embedded-AI/files/hailo_dataflow_compiler_v3.27.0_user_guide.pdf)

### Benchmarks et comparaisons

21. OpenMixup Benchmarks (CUB-200, FGVC-Aircraft). [Documentation](https://openmixup.readthedocs.io/en/latest/mixup_benchmarks/Mixup_downstream.html)
22. Roboflow, *EfficientNet vs. MobileNet V2*. [Roboflow](https://roboflow.com/compare/efficientnet-vs-mobilenet-v2-classification)
23. *Comparative and edge-hybrid modeling of EfficientNetV2 and MobileNetV2*, J. Edge Computing 2025. [ResearchGate](https://www.researchgate.net/publication/396731090)
