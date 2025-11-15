"""
1. Creates/updates TIC subset (60 CP, 90 FP/FA stars)
2. Downloads SPOC light curve FITS files from MAST
3. Converts FITS files to compressed NPZ format
4. Automatically backfills stars without SPOC data
5. Creates final index.csv for training

All steps include resume logic to skip already-processed files.
"""

import pandas as pd
import numpy as np
from astroquery.mast import Observations
from astropy.io import fits
from pathlib import Path
import glob
import random
import os

# Configuration
RANDOM_SEED = 42
TARGET_CP = 60
TARGET_FP = 90
MAX_ITERATIONS = 10

# Paths
FULL_DATASET = 'dataset/toi_per_star_labels.csv'
SUBSET_FILE = 'dataset/tic_subset.csv'
FITS_DIR = Path('dataset/lightcurves_raw/')
NPZ_DIR = Path('dataset/lightcurves_npz/')
INDEX_FILE = 'dataset/index.csv'

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def create_initial_subset():
    """Create initial subset of TICs if it doesn't exist."""
    if Path(SUBSET_FILE).exists():
        print(f"Subset file {SUBSET_FILE} already exists, skipping creation")
        return
    
    print(f"Creating initial subset from {FULL_DATASET}...")
    df = pd.read_csv(FULL_DATASET)
    
    cp_stars = df[df['label'] == 1]
    fp_stars = df[df['label'] == 0]
    
    n_pos = min(TARGET_CP, len(cp_stars))
    n_neg = min(TARGET_FP, len(fp_stars))
    
    if len(cp_stars) < TARGET_CP:
        print(f"Warning: Only {len(cp_stars)} CP stars available, requested {TARGET_CP}")
    if len(fp_stars) < TARGET_FP:
        print(f"Warning: Only {len(fp_stars)} FP/FA stars available, requested {TARGET_FP}")
    
    cp_sample = cp_stars.sample(n=n_pos, random_state=RANDOM_SEED)
    fp_sample = fp_stars.sample(n=n_neg, random_state=RANDOM_SEED)
    
    subset = pd.concat([cp_sample, fp_sample], ignore_index=True)
    subset = subset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    subset.to_csv(SUBSET_FILE, index=False)
    print(f"Created {SUBSET_FILE} with {len(subset)} stars")
    print(f"  - Label 1 (CP): {len(subset[subset['label'] == 1])}")
    print(f"  - Label 0 (FP/FA): {len(subset[subset['label'] == 0])}")


def download_fits_files():
    """Download SPOC light curve FITS files from MAST."""
    print(f"\n{'='*60}")
    print("STEP 1: Downloading FITS files from MAST")
    print(f"{'='*60}")
    
    df = pd.read_csv(SUBSET_FILE)
    FITS_DIR.mkdir(exist_ok=True)
    
    tic_ids = df['tic_id'].unique()
    print(f"Processing {len(tic_ids)} TIC IDs...")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for i, tic_id in enumerate(tic_ids, 1):
        print(f"\n[{i}/{len(tic_ids)}] TIC {tic_id}")
        
        # Check if we already have FITS files for this TIC
        tic_id_str = str(tic_id)
        existing_files = list(FITS_DIR.glob(f"**/*{tic_id_str}*_lc.fits"))
        if len(existing_files) > 0:
            print(f"  Already have {len(existing_files)} FITS file(s), skipping")
            skipped += 1
            continue
        
        try:
            obs_table = Observations.query_criteria(
                objectname=f"TIC {tic_id}",
                obs_collection="TESS",
                dataproduct_type="timeseries"
            )
            
            if len(obs_table) == 0:
                print(f"  No observations found")
                failed += 1
                continue
            
            # Filter observations to only SPOC
            spoc_obs = obs_table[obs_table['provenance_name'] == 'SPOC']
            
            if len(spoc_obs) == 0:
                print(f"  No SPOC observations found")
                failed += 1
                continue
            
            products = Observations.get_product_list(spoc_obs)
            
            # Filter products to only light curves
            lc_indices = []
            for idx, row in enumerate(products):
                if row['productSubGroupDescription'] == 'LC':
                    lc_indices.append(idx)
            
            if len(lc_indices) == 0:
                print(f"  No SPOC light curves found")
                failed += 1
                continue
            
            spoc_lc = products[lc_indices]
            print(f"  Found {len(spoc_lc)} SPOC light curve file(s)")
            
            manifest = Observations.download_products(
                spoc_lc,
                download_dir=str(FITS_DIR)
            )
            
            print(f"  Downloaded {len(manifest)} file(s)")
            downloaded += 1
            
        except Exception as e:
            print(f"  Error: {e}")
            failed += 1
            continue
    
    print(f"\nDownload summary:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Failed/no data: {failed}")


def convert_fits_to_npz():
    """Convert FITS files to compressed NPZ format."""
    print(f"\n{'='*60}")
    print("STEP 2: Converting FITS to NPZ format")
    print(f"{'='*60}")
    
    df = pd.read_csv(SUBSET_FILE)
    NPZ_DIR.mkdir(exist_ok=True)
    
    converted = 0
    skipped = 0
    failed = 0
    
    for tic_id in df['tic_id']:
        tic_id_str = str(tic_id)
        
        # Check if NPZ already exists
        npz_file = NPZ_DIR / f"tic_{tic_id}.npz"
        if npz_file.exists():
            print(f"TIC {tic_id}: NPZ file already exists, skipping")
            skipped += 1
            continue
        
        pattern = str(FITS_DIR / "**" / f"*{tic_id_str}*_lc.fits")
        fits_files = glob.glob(pattern, recursive=True)
        
        if len(fits_files) == 0:
            print(f"TIC {tic_id}: No FITS files found")
            failed += 1
            continue
        
        all_times = []
        all_fluxes = []
        all_flux_errs = []
        all_qualities = []
        all_sectors = []
        all_cameras = []
        all_ccds = []
        all_bkgs = []
        all_crowdsaps = []
        
        for fits_file in sorted(fits_files):
            try:
                with fits.open(fits_file) as hdul:
                    header = hdul[0].header
                    data = hdul[1].data
                    
                    fits_tic = header.get('TICID', None)
                    if fits_tic and str(int(fits_tic)) != tic_id_str:
                        continue
                    
                    sector = header.get('SECTOR', -1)
                    camera = header.get('CAMERA', -1)
                    ccd = header.get('CCD', -1)
                    
                    time = data['TIME'].astype(np.float64)
                    flux = data['PDCSAP_FLUX'].astype(np.float32)
                    flux_err = data['PDCSAP_FLUX_ERR'].astype(np.float32)
                    quality = data['QUALITY'].astype(np.int32)
                    
                    bkg = data['FLUX_BKG'].astype(np.float32) if 'FLUX_BKG' in data.columns.names else None
                    crowdsap = data['CROWDSAP'].astype(np.float32) if 'CROWDSAP' in data.columns.names else None
                    
                    mask = np.isfinite(time) & np.isfinite(flux)
                    
                    all_times.append(time[mask])
                    all_fluxes.append(flux[mask])
                    all_flux_errs.append(flux_err[mask])
                    all_qualities.append(quality[mask])
                    all_sectors.append(np.full(np.sum(mask), sector, dtype=np.int16))
                    all_cameras.append(np.full(np.sum(mask), camera, dtype=np.int8))
                    all_ccds.append(np.full(np.sum(mask), ccd, dtype=np.int8))
                    
                    if bkg is not None:
                        all_bkgs.append(bkg[mask])
                    if crowdsap is not None:
                        all_crowdsaps.append(crowdsap[mask])
                        
            except Exception as e:
                print(f"TIC {tic_id}: Error reading {fits_file}: {e}")
                continue
        
        if len(all_times) == 0:
            print(f"TIC {tic_id}: No valid data extracted")
            failed += 1
            continue
        
        t = np.concatenate(all_times)
        flux = np.concatenate(all_fluxes)
        flux_err = np.concatenate(all_flux_errs)
        quality = np.concatenate(all_qualities)
        sector = np.concatenate(all_sectors)
        camera = np.concatenate(all_cameras)
        ccd = np.concatenate(all_ccds)
        
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        flux = flux[sort_idx]
        flux_err = flux_err[sort_idx]
        quality = quality[sort_idx]
        sector = sector[sort_idx]
        camera = camera[sort_idx]
        ccd = ccd[sort_idx]
        
        save_dict = {
            't': t,
            'flux': flux,
            'flux_err': flux_err,
            'quality': quality,
            'sector': sector,
            'camera': camera,
            'ccd': ccd
        }
        
        if len(all_bkgs) > 0:
            bkg = np.concatenate(all_bkgs)[sort_idx]
            save_dict['bkg'] = bkg
        
        if len(all_crowdsaps) > 0:
            crowdsap = np.concatenate(all_crowdsaps)[sort_idx]
            save_dict['crowdsap'] = crowdsap
        
        np.savez_compressed(npz_file, **save_dict)
        print(f"TIC {tic_id}: Saved {len(t)} cadences from {len(fits_files)} file(s)")
        converted += 1
    
    print(f"\nConversion summary:")
    print(f"  Converted: {converted}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Failed/no data: {failed}")


def check_status():
    """Check current status and return counts."""
    subset_df = pd.read_csv(SUBSET_FILE)
    npz_files = list(NPZ_DIR.glob('tic_*.npz'))
    npz_tic_ids = {int(f.stem.split('_')[1]) for f in npz_files}
    
    cp_with_npz = len(subset_df[(subset_df['label'] == 1) & (subset_df['tic_id'].isin(npz_tic_ids))])
    fp_with_npz = len(subset_df[(subset_df['label'] == 0) & (subset_df['tic_id'].isin(npz_tic_ids))])
    
    return cp_with_npz, fp_with_npz, npz_tic_ids


def backfill_subset(iteration=0):
    """Replace stars without SPOC data with new random selections."""
    print(f"\n{'='*60}")
    print(f"STEP 3: Backfilling subset (iteration {iteration + 1})")
    print(f"{'='*60}")
    
    full_df = pd.read_csv(FULL_DATASET)
    subset_df = pd.read_csv(SUBSET_FILE)
    npz_tic_ids = check_status()[2]
    
    current_cp, current_fp, _ = check_status()
    
    print(f"Current status:")
    print(f"  CP with NPZ: {current_cp} / {TARGET_CP}")
    print(f"  FP/FA with NPZ: {current_fp} / {TARGET_FP}")
    
    if current_cp >= TARGET_CP and current_fp >= TARGET_FP:
        print("Target already reached!")
        return True
    
    subset_tic_ids = set(subset_df['tic_id'])
    missing_tic_ids = subset_tic_ids - npz_tic_ids
    
    cp_pool = full_df[(full_df['label'] == 1) & (~full_df['tic_id'].isin(subset_tic_ids))]
    fp_pool = full_df[(full_df['label'] == 0) & (~full_df['tic_id'].isin(subset_tic_ids))]
    
    new_cp_needed = max(0, TARGET_CP - current_cp)
    new_fp_needed = max(0, TARGET_FP - current_fp)
    
    print(f"\nNeed to add:")
    print(f"  CP: {new_cp_needed}")
    print(f"  FP/FA: {new_fp_needed}")
    
    if new_cp_needed > len(cp_pool) or new_fp_needed > len(fp_pool):
        print(f"Cannot reach target: insufficient stars in pool")
        print(f"  CP pool: {len(cp_pool)}, need: {new_cp_needed}")
        print(f"  FP pool: {len(fp_pool)}, need: {new_fp_needed}")
        return False
    
    new_cp = cp_pool.sample(n=new_cp_needed, random_state=RANDOM_SEED + iteration) if new_cp_needed > 0 else pd.DataFrame()
    new_fp = fp_pool.sample(n=new_fp_needed, random_state=RANDOM_SEED + iteration) if new_fp_needed > 0 else pd.DataFrame()
    
    valid_subset = subset_df[subset_df['tic_id'].isin(npz_tic_ids)].copy()
    new_stars = pd.concat([new_cp, new_fp], ignore_index=True) if len(new_cp) > 0 or len(new_fp) > 0 else pd.DataFrame()
    
    if len(new_stars) > 0:
        updated_subset = pd.concat([valid_subset, new_stars], ignore_index=True)
        updated_subset = updated_subset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        updated_subset.to_csv(SUBSET_FILE, index=False)
        
        print(f"\nUpdated subset:")
        print(f"  Added {len(new_stars)} new stars ({len(new_cp)} CP, {len(new_fp)} FP/FA)")
        print(f"  Removed {len(missing_tic_ids)} stars without NPZ files")
        print(f"  New total: {len(updated_subset)} stars")
        return False  # Need to continue
    else:
        valid_subset = valid_subset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        valid_subset.to_csv(SUBSET_FILE, index=False)
        print(f"\nCleaned up subset (removed {len(missing_tic_ids)} stars without NPZ files)")
        return current_cp >= TARGET_CP and current_fp >= TARGET_FP


def create_index():
    """Create final index.csv file for training."""
    print(f"\n{'='*60}")
    print("STEP 4: Creating index.csv")
    print(f"{'='*60}")
    
    subset_df = pd.read_csv(SUBSET_FILE)
    npz_files = list(NPZ_DIR.glob('tic_*.npz'))
    npz_tic_ids = {int(f.stem.split('_')[1]) for f in npz_files}
    
    index_rows = []
    
    for _, row in subset_df.iterrows():
        tic_id = row['tic_id']
        
        if tic_id in npz_tic_ids:
            npz_file = next((f for f in npz_files if int(f.stem.split('_')[1].replace('.0', '')) == tic_id), None)
            if npz_file:
                npz_path = f"lightcurves_npz/{npz_file.name}"
            else:
                npz_path = f"lightcurves_npz/tic_{tic_id}.npz"
            
            index_rows.append({
                'tic_id': int(tic_id),
                'npz_path': npz_path,
                'label': int(row['label']),
                'period_days': row['period_days'] if pd.notna(row['period_days']) else None,
                't0_bjd': row['t0_bjd'] if pd.notna(row['t0_bjd']) else None
            })
    
    index_df = pd.DataFrame(index_rows).sort_values('tic_id').reset_index(drop=True)
    index_df.to_csv(INDEX_FILE, index=False)
    
    print(f"Created {INDEX_FILE} with {len(index_df)} entries")
    print(f"  - Label 1 (CP): {len(index_df[index_df['label'] == 1])}")
    print(f"  - Label 0 (FP/FA): {len(index_df[index_df['label'] == 0])}")
    print(f"  - TICs in subset but no NPZ: {len(subset_df) - len(index_df)}")


def main():
    """Main pipeline execution."""
    print("="*60)
    print("TESS Light Curve Dataset Builder")
    print("="*60)
    
    # Create initial subset if needed
    create_initial_subset()
    
    # Iterative pipeline until targets are met
    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'#'*60}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'#'*60}")
        
        # Download FITS files
        download_fits_files()
        
        # Convert to NPZ
        convert_fits_to_npz()
        
        # Check status and backfill if needed
        current_cp, current_fp, _ = check_status()
        
        if current_cp >= TARGET_CP and current_fp >= TARGET_FP:
            print(f"\n[SUCCESS] Target reached: {current_cp} CP, {current_fp} FP/FA")
            break
        
        if not backfill_subset(iteration):
            print(f"\nContinuing to next iteration...")
            continue
        else:
            break
    
    # Create final index
    create_index()
    
    # Final status
    current_cp, current_fp, _ = check_status()
    print(f"\n{'='*60}")
    print("FINAL STATUS")
    print(f"{'='*60}")
    print(f"CP: {current_cp} / {TARGET_CP}")
    print(f"FP/FA: {current_fp} / {TARGET_FP}")
    print(f"Total with NPZ: {current_cp + current_fp}")
    
    if current_cp >= TARGET_CP and current_fp >= TARGET_FP:
        print("\n[SUCCESS] Successfully completed dataset building!")
    else:
        print(f"\n[WARNING] Could not reach target after {MAX_ITERATIONS} iterations")


if __name__ == '__main__':
    main()

