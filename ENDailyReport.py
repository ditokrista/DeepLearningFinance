import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# DATA TRANSFORMATION TO TIDY FORMAT
# ==========================================

# Read the raw CSV with multi-level headers
df_raw = pd.read_csv("C:/Users/PC/Documents/1. Coal Hauling Project/2. Operations/Daily Report Dashboard/EN Daily Report.csv", 
                     header=[0, 1])

# Identify key column indices
# Static columns: No., Jenis DT, No. Lambung, Kategori, Site (0-4)
# Date data starts at column 5, ends at column 278 (before Status DT Terakhir at 279)
# Status DT Terakhir columns: 279-280 (Keterangan, Action) - we'll drop these

print("=" * 80)
print("STEP 1: Extract static truck information")
print("=" * 80)

# Flatten column names for easier handling
df_raw.columns = ['_'.join(map(str, col)).strip() for col in df_raw.columns]

# Extract static columns (truck info)
static_cols = df_raw.iloc[:, :5].copy()
static_cols.columns = ['No', 'Jenis_DT', 'No_Lambung', 'Kategori', 'Site']

print(f"Static columns shape: {static_cols.shape}")
print(static_cols.head(3))

print("\n" + "=" * 80)
print("STEP 2: Extract and reshape date-based route data")
print("=" * 80)

# Extract only date-related columns (5 to 278)
date_data = df_raw.iloc[:, 5:279].copy()
print(f"Date data columns: {date_data.shape[1]}")

# Identify the pattern of columns
# Each date block has 7 columns: 107-Port (R,T), BSR-107 (R,T), BA (R,T), Keterangan
# We need to reshape this into long format

# Create list to store all transformed records
all_records = []

# Process each truck (row)
for truck_idx in range(len(static_cols)):
    truck_info = static_cols.iloc[truck_idx]
    
    # Process date blocks (each block = 7 columns)
    date_idx = 0
    col_idx = 0
    
    while col_idx < date_data.shape[1] - 6:  # Ensure we have at least 7 columns left
        # Generate date (October 1-31, 2025)
        date = pd.Timestamp('2025-10-01') + pd.Timedelta(days=date_idx)
        
        # Stop if we exceed October
        if date.month > 10:
            break
            
        # Extract data for this date block
        try:
            ritase_107port = date_data.iloc[truck_idx, col_idx]
            tonase_107port = date_data.iloc[truck_idx, col_idx + 1]
            ritase_bsr107 = date_data.iloc[truck_idx, col_idx + 2]
            tonase_bsr107 = date_data.iloc[truck_idx, col_idx + 3]
            ritase_ba = date_data.iloc[truck_idx, col_idx + 4]
            tonase_ba = date_data.iloc[truck_idx, col_idx + 5]
            keterangan = date_data.iloc[truck_idx, col_idx + 6]
            
            # Create record for 107-Port route
            if pd.notna(ritase_107port) or pd.notna(tonase_107port):
                all_records.append({
                    'Date': date,
                    'No': truck_info['No'],
                    'Jenis_DT': truck_info['Jenis_DT'],
                    'No_Lambung': truck_info['No_Lambung'],
                    'Kategori': truck_info['Kategori'],
                    'Site': truck_info['Site'],
                    'Route': '107-Port',
                    'Ritase': ritase_107port,
                    'Tonase': tonase_107port,
                    'Keterangan': keterangan
                })
            
            # Create record for BSR-107 route
            if pd.notna(ritase_bsr107) or pd.notna(tonase_bsr107):
                all_records.append({
                    'Date': date,
                    'No': truck_info['No'],
                    'Jenis_DT': truck_info['Jenis_DT'],
                    'No_Lambung': truck_info['No_Lambung'],
                    'Kategori': truck_info['Kategori'],
                    'Site': truck_info['Site'],
                    'Route': 'BSR-107',
                    'Ritase': ritase_bsr107,
                    'Tonase': tonase_bsr107,
                    'Keterangan': keterangan
                })
            
        except IndexError:
            break
            
        col_idx += 7
        date_idx += 1

print(f"\nTotal records created: {len(all_records)}")

print("\n" + "=" * 80)
print("STEP 3: Create tidy dataframe")
print("=" * 80)

# Create the tidy dataframe
df_tidy = pd.DataFrame(all_records)

# Reorder columns for clarity
column_order = ['Date', 'No', 'Jenis_DT', 'No_Lambung', 'Kategori', 'Site', 
                'Route', 'Ritase', 'Tonase', 'Keterangan']
df_tidy = df_tidy[column_order]

# Sort by Date, then by truck number
df_tidy = df_tidy.sort_values(['Date', 'No', 'Route']).reset_index(drop=True)

print(f"Tidy dataframe shape: {df_tidy.shape}")
print(f"\nFirst 10 rows:")
print(df_tidy.head(10))
print(f"\nData types:")
print(df_tidy.dtypes)
print(f"\nDate range: {df_tidy['Date'].min()} to {df_tidy['Date'].max()}")
print(f"Unique routes: {df_tidy['Route'].unique()}")

print("\n" + "=" * 80)
print("STEP 4: Save cleaned data")
print("=" * 80)

# Save to new CSV
output_path = "C:/Users/PC/Documents/1. Coal Hauling Project/2. Operations/Daily Report Dashboard/EN Daily Report_CLEANED.csv"
df_tidy.to_csv(output_path, index=False)
print(f"[SUCCESS] Cleaned data saved to: {output_path}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"• Original shape: {df_raw.shape}")
print(f"• Cleaned shape: {df_tidy.shape}")
print(f"• Removed 'Status DT Terakhir' columns (Keterangan, Action)")
print(f"• Melted '107-Port' and 'BSR-107' into 'Route' column")
print(f"• Added 'Date' column starting from 2025-10-01")
print(f"• Data follows tidy data principles: each row = 1 observation")

print("\n" + "=" * 80)
print("DATA QUALITY CHECKS")
print("=" * 80)

# Check for data completeness
print(f"\nMissing values per column:")
print(df_tidy.isnull().sum())

print(f"\nRecords per route:")
print(df_tidy['Route'].value_counts())

print(f"\nRecords per date (first 10 dates):")
print(df_tidy['Date'].value_counts().sort_index().head(10))

print(f"\nSample of cleaned data (random 15 rows):")
print(df_tidy.sample(min(15, len(df_tidy)), random_state=42).to_string())

print("\n" + "=" * 80)
print("TRANSFORMATION COMPLETE!")
print("=" * 80)

