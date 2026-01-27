# Emotion Recognition - Projet Éducatif

> **Note : Ce projet est à des fins éducatives uniquement**

Un projet de reconnaissance d'émotions faciales en temps réel utilisant PyTorch et OpenCV. Ce projet démontre les concepts fondamentaux du deep learning appliqués à la vision par ordinateur, incluant la préparation de données, la création de datasets personnalisés, et la détection de visages en temps réel.

## 📋 Description

Ce projet implémente un système de reconnaissance d'émotions faciales capable de :
- Charger et préprocesser un dataset d'expressions faciales (FER)
- Créer un dataset PyTorch personnalisé
- Détecter des visages en temps réel via webcam
- Extraire et normaliser les régions faciales pour la classification

## 🏗️ Structure du Projet

```
emotion_rt/
├── data/
│   └── train/              # Dataset d'entraînement organisé par classes
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
├── fer_dataset.py          # Dataset PyTorch personnalisé
├── build_index.py          # Construction de l'index et split train/val
├── dataset_preview.py      # Visualisation et exploration du dataset
├── realtime_cam.py         # Détection de visages en temps réel
└── test_dataloader.py      # Test du DataLoader PyTorch
```

## 🚀 Installation

### Prérequis

- Python 3.8+
- Webcam (pour la détection en temps réel)

### Dépendances

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
```

## 📚 Utilisation

### 1. Exploration du Dataset

Visualiser les classes disponibles et quelques échantillons :

```bash
python dataset_preview.py
```

### 2. Construction de l'Index

Construire l'index des échantillons et diviser en train/validation :

```bash
python build_index.py
```

Cette commande affiche :
- Le mapping classe → ID
- La répartition train/validation
- Des exemples d'échantillons

### 3. Test du DataLoader

Vérifier que le DataLoader fonctionne correctement :

```bash
python test_dataloader.py
```

### 4. Détection de Visages en Temps Réel

Lancer la détection de visages via webcam :

```bash
python realtime_cam.py
```

**Contrôles :**
- Appuyez sur `q` pour quitter
- Les visages détectés sont encadrés en vert
- Le visage extrait (48×48) est affiché en haut à droite

## 🔍 Détails Techniques

### Dataset (`fer_dataset.py`)

La classe `FerDataset` hérite de `torch.utils.data.Dataset` et implémente :
- `__len__()` : Retourne le nombre d'échantillons
- `__getitem__()` : Charge et préprocesse une image
  - Conversion en niveaux de gris
  - Redimensionnement à 48×48 pixels
  - Normalisation [0, 1]
  - Ajout d'une dimension de canal

### Construction de l'Index (`build_index.py`)

- `build_samples()` : Parcourt le répertoire `data/train/` et construit une liste de tuples `(chemin_image, id_classe)`
- `split_samples()` : Divise les données en train/validation de manière reproductible (seed=42)

### Détection Temps Réel (`realtime_cam.py`)

- Utilise le classificateur Haar Cascade d'OpenCV pour détecter les visages
- Extrait le visage le plus grand dans le frame
- Normalise l'extraction à 48×48 pixels (format attendu par le modèle)

## 📖 Concepts Éducatifs

Ce projet illustre plusieurs concepts importants :

1. **Datasets PyTorch** : Création d'un dataset personnalisé héritant de `torch.utils.data.Dataset`
2. **DataLoader** : Utilisation de `torch.utils.data.DataLoader` pour le chargement par batch
3. **Préprocessing** : Normalisation et redimensionnement d'images
4. **Détection d'objets** : Utilisation de Haar Cascades pour la détection de visages
5. **Traitement vidéo** : Capture et traitement de flux vidéo en temps réel avec OpenCV

## 🎯 Prochaines Étapes (Suggestions)

Pour étendre ce projet, vous pourriez :

1. **Entraîner un modèle** : Créer un réseau de neurones (CNN) pour classifier les émotions
2. **Intégration** : Combiner `realtime_cam.py` avec un modèle entraîné pour prédire les émotions en temps réel
3. **Amélioration de la détection** : Utiliser MTCNN ou MediaPipe pour une meilleure détection de visages
4. **Augmentation de données** : Implémenter des transformations (rotation, flip, etc.) pour améliorer la robustesse
5. **Métriques** : Ajouter des métriques d'évaluation (accuracy, confusion matrix, etc.)

## ⚠️ Avertissement

Ce projet est conçu à des fins **éducatives uniquement**. Pour une utilisation en production, considérez :
- La qualité et la diversité du dataset
- Les biais potentiels dans les données
- Les aspects éthiques de la reconnaissance d'émotions
- Les performances et l'optimisation du modèle

## 📝 Notes

- Le dataset doit être organisé dans `data/train/` avec un dossier par classe d'émotion
- Les images supportées sont : `.jpg`, `.jpeg`, `.png`
- Les images sont automatiquement redimensionnées à 48×48 pixels si nécessaire
- La détection de visages utilise le classificateur Haar Cascade par défaut d'OpenCV

## 📄 Licence

Projet éducatif - Utilisation libre pour l'apprentissage.
