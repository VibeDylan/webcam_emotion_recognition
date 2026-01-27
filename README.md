# Emotion Recognition - Real-time

Projet de détection d'émotions en temps réel avec une webcam, utilisant un CNN entraîné sur le dataset FER2013.

## Description

Ce projet permet de détecter les émotions faciales en temps réel via la webcam. Le modèle est un CNN entraîné sur 7 classes d'émotions : angry, disgust, fear, happy, neutral, sad, surprise.

## Structure du projet

```
emotion_rt/
├── src/
│   ├── models/
│   │   └── emotion_cnn.py          # Architecture du modèle CNN
│   ├── data/
│   │   ├── dataset_utils.py         # Utilitaires pour le dataset
│   │   ├── build_index.py          # Construction de l'index et split train/val
│   │   └── fer_dataset.py          # Dataset PyTorch pour le chargement des images
│   └── utils/
│       ├── model_loader.py         # Fonctions pour charger le modèle entraîné
│       └── prediction.py           # Fonction de prédiction d'émotion
├── train.py                         # Script d'entraînement du modèle
├── realtime_cam.py                  # Application de détection temps réel
└── emotion_best.pt                  # Modèle entraîné (sauvegardé automatiquement)
```

## Installation

Assure-toi d'avoir installé les dépendances :
- torch
- torchvision
- opencv-python
- numpy

## Utilisation

### Entraînement du modèle

Pour entraîner le modèle sur le dataset FER2013 :

```bash
python train.py
```

Le script va :
- Charger et préparer les données (split train/val 90/10)
- Entraîner le modèle pendant 30 epochs
- Sauvegarder automatiquement le meilleur modèle dans `emotion_best.pt`
- Afficher la progression avec les métriques à chaque epoch

### Détection en temps réel

Pour lancer la détection d'émotions en temps réel avec la webcam :

```bash
python realtime_cam.py
```

L'application va :
- Charger le modèle entraîné
- Détecter les visages dans le flux vidéo
- Prédire l'émotion avec un lissage sur 10 frames (anti-flicker)
- Afficher l'émotion et la confiance sur la vidéo

Appuie sur `q` pour quitter.

## Architecture du modèle

Le modèle utilise une architecture CNN avec :
- 4 couches de convolution avec Batch Normalization
- 3 couches fully connected
- Dropout pour la régularisation
- Max pooling pour réduire la dimensionnalité

## Notes

Le modèle est entraîné sur des images 48x48 en niveaux de gris. Les visages détectés sont automatiquement redimensionnés à cette taille avant la prédiction.
