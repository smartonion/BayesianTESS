import pandas as pd
import numpy as np
from pathlib import Path
import random

df = pd.read_csv('dataset/tic_subset.csv')
npz_dir = Path('dataset/lightcurves_npz/')

npz_files = list(npz_dir.glob('tic_*.npz'))
npz_tic_ids = {int(f.stem.split('_')[1]) for f in npz_files}

print(f"TICs in subset: {len(df)}")
print(f"TICs with NPZ files: {len(npz_tic_ids)}")
print(f"Missing NPZ files: {len(df) - len(npz_tic_ids)}")

if len(npz_files) == 0:
    print("\nNo NPZ files found. Run build_npz_from_fits.py first.")
    exit()

total_size = sum(f.stat().st_size for f in npz_files)
print(f"\nTotal size of lightcurves_npz/: {total_size / (1024**3):.2f} GB")

sectors_per_star = []
sample_size = min(10, len(npz_files))
sample_files = random.sample(npz_files, sample_size)

print(f"\nSample inspection ({sample_size} random stars):")
print("-" * 60)

for npz_file in sorted(sample_files):
    data = np.load(npz_file)
    tic_id = int(npz_file.stem.split('_')[1])
    
    t = data['t']
    flux = data['flux']
    sectors = np.unique(data['sector'])
    
    print(f"\nTIC {tic_id}:")
    print(f"  Cadences: {len(t):,}")
    print(f"  Sectors: {sorted(sectors)}")
    print(f"  Time range: {t.min():.2f} - {t.max():.2f} BJD")
    print(f"  Median flux: {np.median(flux):.2f}")
    
    sectors_per_star.append(len(sectors))

print("\n" + "=" * 60)
print("Summary statistics:")
print(f"  Stars with at least one sector: {len(npz_tic_ids)}")
print(f"  Sectors per star:")
print(f"    Min: {min(sectors_per_star)}")
print(f"    Max: {max(sectors_per_star)}")
print(f"    Median: {np.median(sectors_per_star):.0f}")
print(f"    Mean: {np.mean(sectors_per_star):.1f}")

