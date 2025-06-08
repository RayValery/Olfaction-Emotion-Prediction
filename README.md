# Olfaction-Emotion-Prediction

This project explores how molecular features of odorants can be used to predict perceptual descriptors such as "sweet", "musky", or "fruity", using machine learning.

## 🔬 Project Goals

- Predict odor descriptors (e.g., SWEET) from molecular features.
- Explore how chemical structure relates to emotional and perceptual responses.
- Lay the groundwork for future research on emotion-linked smell perception.

## ✅ Current Progress

- [x] Baseline classification model (Random Forest) for predicting SWEET.
- [ ] Multi-label classification (predicting multiple descriptors).
- [ ] Emotion prediction (pleasantness, arousal).
- [ ] Clustering of odors by perceptual similarity.

## 📁 Structure

```
src/
  └── baseline_sweet_prediction.py  # Current working model
data/
  ├── TrainSet.txt
  ├── molecular_descriptors_data.txt
  └── ...
```

## 🚀 How to run

```bash
# Setup
pip install -r requirements.txt

# Run the baseline
python src/baseline_sweet_prediction.py
```

## 📦 Requirements

- Python 3.8+
- pandas
- scikit-learn

(Optionally: matplotlib, seaborn for visualizations)

---

## 🧠 Author

Valerie Salivon — exploring the intersection of machine learning, olfaction, and affective neuroscience.

---

## 🌱 Future Ideas

- Predict emotional response from odor.
- Visualize latent odor space.
- Simulate brain-like olfactory responses using GNNs.
