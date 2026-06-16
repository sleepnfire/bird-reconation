# Démo Webcam — Guide d'utilisation

Script : `demo_webcam.py`

---

## Prérequis

- Python 3.14 avec l'environnement virtuel activé (`source .venv/bin/activate`)
- OpenCV (`pip install opencv-python`)
- PyTorch + torchvision
- Une webcam connectée

---

## Mode simple — un seul modèle

```bash
# Charge le meilleur modèle du dernier run
python demo_webcam.py

# Charge le meilleur modèle d'un run spécifique
python demo_webcam.py 1
python demo_webcam.py 2
python demo_webcam.py 3

# Charge un checkpoint précis
python demo_webcam.py output/3/best_vit_b_16.pth
python demo_webcam.py output/1/best_mobilenetv2.pth
```

Le script détecte automatiquement l'architecture depuis le checkpoint. Quand on donne un numéro de run, il choisit le modèle le plus puissant selon ce classement :

```
ViT-B/16 > EfficientNetV2-B2 > EfficientNet-B0 > MobileNetV2
```

---

## Mode comparaison — écran divisé

### Comparer le meilleur de chaque run (entre runs)

```bash
# Les 2 à 4 derniers runs côte à côte (meilleur modèle de chaque)
python demo_webcam.py --compare

# Deux runs spécifiques
python demo_webcam.py --compare 1 3

# Trois runs spécifiques
python demo_webcam.py --compare 1 2 3
```

Exemple avec `--compare` (3 runs disponibles) :

```
┌───────────────────┬────────────────────┬──────────────────┐
│ Run 1             │ Run 2              │ Run 3            │
│ MobileNetV2       │ EfficientNet-B0    │ ViT-B/16         │
│                   │                    │                  │
│ Mésange  35%      │ Mésange  52%       │ Mésange  85%     │
│ Pinson   20%      │ Pinson   18%       │ Pinson   5%      │
│ ...               │ ...                │ ...              │
└───────────────────┴────────────────────┴──────────────────┘
```

### Comparer les modèles d'un même run (intra-run)

```bash
# Tous les modèles du run 3 côte à côte
python demo_webcam.py 3 --compare
```

Exemple avec le run 3 (3 modèles) :

```
┌──────────────────┬──────────────────┬──────────────────┐
│ ViT-B/16         │ EfficientNet-B0  │ MobileNetV2      │
│                  │                  │                  │
│ Mésange  85%     │ Mésange  62%     │ Pinson   41%     │
│ Pinson   5%      │ Pinson   18%     │ Mésange  35%     │
│ ...              │ ...              │ ...              │
└──────────────────┴──────────────────┴──────────────────┘
```

### Comparer des checkpoints spécifiques

```bash
python demo_webcam.py --compare output/1/best_mobilenetv2.pth output/3/best_mobilenetv2.pth
```

---

## Contrôles

| Touche | Action |
|--------|--------|
| **Q** | Quitter |

---

## Modèles disponibles par run

| Run | Modèles | val_acc du meilleur |
|:---:|---------|:-------------------:|
| 1 | MobileNetV2 | 62.0 % |
| 2 | EfficientNet-B0, MobileNetV2 | — |
| 3 | ViT-B/16, EfficientNet-B0, MobileNetV2 | 79.5 % |

---

## Notes

- L'inférence tourne sur le GPU Apple Silicon (MPS) si disponible, sinon CPU
- La prédiction est faite toutes les 3 frames pour garder la fluidité
- Le top 5 des espèces est affiché avec le nom français et le pourcentage de confiance
- Barres vertes (>60 % de confiance), orange (30-60 %), grises (prédictions secondaires)
- Le modèle a été entraîné sur des photos d'oiseaux — pour un vrai test, montrer une photo d'oiseau à la caméra
