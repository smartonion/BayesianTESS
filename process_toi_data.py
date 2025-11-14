import pandas as pd

df = pd.read_csv('dataset/TOI_2025.11.14_12.44.03.csv', comment='#')

df = df[df['tid'].notna()]
df = df[~df['tfopwg_disp'].isin(['PC', 'APC', 'KP'])]

# Group by TIC ID: label=1 if has CP, label=0 if only FP/FA, skip ambiguous stars
result_rows = []

for tic_id, group in df.groupby('tid'):
    dispositions = set(group['tfopwg_disp'].dropna().unique())
    has_cp = 'CP' in dispositions
    has_fp_fa = bool(dispositions & {'FP', 'FA'})
    
    if has_cp and has_fp_fa:
        continue
    
    if has_cp:
        # For multiple CP TOIs, select the one with lowest TOI number
        cp_rows = group[group['tfopwg_disp'] == 'CP'].copy()
        cp_rows = cp_rows.sort_values('toi')
        first_cp = cp_rows.iloc[0]
        
        result_rows.append({
            'tic_id': int(tic_id),
            'label': 1,
            'period_days': first_cp['pl_orbper'] if pd.notna(first_cp['pl_orbper']) else None,
            't0_bjd': first_cp['pl_tranmid'] if pd.notna(first_cp['pl_tranmid']) else None
        })
    
    elif has_fp_fa:
        result_rows.append({
            'tic_id': int(tic_id),
            'label': 0,
            'period_days': None,
            't0_bjd': None
        })

result_df = pd.DataFrame(result_rows).sort_values('tic_id').reset_index(drop=True)

output_file = 'dataset/toi_per_star_labels.csv'
result_df.to_csv(output_file, index=False)

print(f"Final result: {len(result_df)} stars")
print(f"  - Label 1 (CP): {len(result_df[result_df['label'] == 1])}")
print(f"  - Label 0 (FP/FA): {len(result_df[result_df['label'] == 0])}")
print(f"Saved to {output_file}")

