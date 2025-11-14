# BayesianTESS
Applying Bayesian Methods in Machine Learning on TESS Light Curves

## What's been done

Added the TOI dataset from NASA Exoplanet Archive. Created a processing script that converts the raw TOI data into a clean format with one row per star. The script filters out ambiguous cases and dispositions we don't want (PC, APC, KP), and creates binary labels: 1 for confirmed planets, 0 for false positives/alarms. Output is saved to `dataset/toi_per_star_labels.csv` with columns: tic_id, label, period_days, t0_bjd. This is the file that'll be used for the GP training pipeline.