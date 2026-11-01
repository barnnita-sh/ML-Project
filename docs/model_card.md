# Model card

- **Model A** (pickups per hexagon-hour): Boosted trees (Poisson), setting {'learning_rate': 0.05, 'max_leaf_nodes': 127, 'min_samples_leaf': 50, 'l2_regularization': 1.0, 'max_features': 0.5}.
  Validation Poisson deviance 0.757 (historical average 0.782).
- **Model B** (minutes until pickup): Boosted trees (log time), setting {'learning_rate': 0.03, 'max_leaf_nodes': 63, 'min_samples_leaf': 500, 'l2_regularization': 10.0, 'max_features': 1.0}.
  Validation C-index 0.604 (historical curves 0.574), Brier at 15 min 0.230.
- **Fallback** for Model B: historical survival curves by area, 3-hour band and minutes-empty band.
- **Trained on** 29 Jan to 15 Sep 2025 (29 Aug excluded), sampled as in `model_input/info.json`.
- **Inputs**: the features in `docs/features.md`; Model B also uses minutes empty.
- **Not for**: claims about real driver time savings; areas and hours with little data (use the coverage rule).
- Files: `pipeline_output/models/*.joblib`, choice in `pipeline_output/models/final.json`.
