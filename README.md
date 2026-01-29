# Emotion Recognition - Real-time

Projet de détection d'émotions en temps réel avec une webcam, utilisant un CNN ou un ResNet entraîné sur un dataset FER (7 classes).

## Description

Ce projet permet de détecter les émotions faciales en temps réel via la webcam. Deux modèles sont disponibles : un **CNN** et un **ResNet18** (optionnellement pré-entraîné sur ImageNet). Les deux sont entraînés sur 7 classes d'émotions : angry, disgust, fear, happy, neutral, sad, surprise.

## Structure du projet

```
emotion_rt/
├── src/
│   ├── models/
│   │   ├── emotion_cnn.py          # Architecture CNN
│   │   └── emotion_resnet.py       # Architecture ResNet18 (1 canal, 7 classes)
│   ├── data/
│   │   ├── dataset_utils.py       # Chemins et utilitaires dataset
│   │   ├── build_index.py         # Index et split train/val
│   │   └── fer_dataset.py         # Dataset PyTorch (chargement images)
│   └── utils/
│       ├── model_loader.py        # Chargement du modèle (CNN ou ResNet)
│       └── prediction.py          # Prédiction d'émotion (face 48x48)
├── data/
│   └── train/                     # Images par classe (un dossier par émotion)
├── train.py                        # Entraînement (CNN ou ResNet)
├── realtime_cam.py                 # Détection temps réel webcam
├── emotion_best.pt                 # Modèle CNN (généré par train.py)
├── emotion_resnet_best.pt          # Modèle ResNet (généré par train.py)
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Les données d'entraînement doivent être dans `data/train/` avec un sous-dossier par classe (ex. `data/train/angry/`, `data/train/happy/`, etc.).

## Utilisation

### Entraînement

**CNN (défaut) :**
```bash
python train.py
# ou explicitement :
python train.py --model cnn
```
Sauvegarde du meilleur modèle dans `emotion_best.pt`.

**ResNet (sans poids ImageNet) :**
```bash
python train.py --model resnet
```

**ResNet avec poids pré-entraînés ImageNet (recommandé) :**
```bash
python train.py --model resnet --pretrained
```
Sauvegarde du meilleur modèle dans `emotion_resnet_best.pt`.

Comportement de l'entraînement :
- Split train/val 90/10, batch_size 64
- Jusqu'à 50 epochs avec **early stopping** (arrêt si pas d'amélioration de l'accuracy val pendant 10 epochs)
- **ReduceLROnPlateau** sur l'accuracy de validation (factor=0.5, patience=3)
- Sauvegarde automatique du meilleur modèle selon l'accuracy de validation

### Détection en temps réel

**Avec le CNN :**
```bash
python realtime_cam.py
# ou :
python realtime_cam.py --model cnn
```

**Avec le ResNet :**
```bash
python realtime_cam.py --model resnet
```

L'application charge le modèle correspondant (`emotion_best.pt` ou `emotion_resnet_best.pt`), détecte les visages, prédit l'émotion avec un lissage sur 10 frames et affiche le label + confiance. Appuie sur **q** pour quitter.

## Modèles

- **CNN** : 4 blocs convolution + BatchNorm, couches fully connected, dropout.
- **ResNet18** : ResNet18 adapté en entrée 1 canal (grayscale), 7 classes ; option `--pretrained` pour initialiser avec ImageNet (conv1 adapté par moyenne des canaux).

Les entrées sont des images **48×48 en niveaux de gris**. Les visages détectés par la webcam sont recadrés et redimensionnés à cette taille avant prédiction.
