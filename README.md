# Emotion Recognition - Real-time

Détection d'émotions en temps réel avec CNN.

## Structure

```
emotion_rt/
├── src/
│   ├── models/
│   │   └── emotion_cnn.py          # Modèle CNN
│   ├── data/
│   │   ├── dataset_utils.py        # Utilitaires dataset
│   │   ├── build_index.py          # Construction index et split
│   │   └── fer_dataset.py          # Dataset PyTorch
│   └── utils/
│       ├── model_loader.py          # Chargement modèle
│       └── prediction.py            # Prédiction émotion
├── train.py                         # Script d'entraînement
├── realtime_cam.py                  # Application temps réel
└── emotion_best.pt                  # Modèle entraîné

```

## Usage

### Entraînement
```bash
python train.py
```

### Détection temps réel
```bash
python realtime_cam.py
```
