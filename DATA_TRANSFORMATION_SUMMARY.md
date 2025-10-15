# EN Daily Report - Data Transformation Summary

## Overview
Transformed the EN Daily Report CSV from a wide, multi-observation format into a **tidy data format** following data science best practices.

## Transformations Applied

### 1. **Route Melting** ✓
- **Before**: Routes `107-Port` and `BSR-107` were spread across multiple columns
- **After**: Melted into a single `Route` column with corresponding `Ritase` and `Tonase` values
- **Result**: Each row now represents one route observation per truck per date

### 2. **Date Column Creation** ✓
- **Added**: `Date` column starting from **October 1, 2025**
- **Range**: October 1-31, 2025 (31 days)
- **Type**: `datetime64[ns]` for proper temporal analysis

### 3. **Removed Status Columns** ✓
- **Deleted**: Last 2 columns under "Status DT Terakhir":
  - `Keterangan` (status)
  - `Action`
- These columns were metadata not relevant for data analysis

### 4. **Tidy Data Principles** ✓
- Each row = **1 observation** (single truck, single route, single date)
- Each column = **1 variable**
- Each cell = **1 value**

## Data Structure

### Original Format
- **Shape**: 251 rows × 330 columns
- **Issues**:
  - Multiple observations per row (different routes, different dates)
  - Column headers spread across multiple rows
  - Wide format made analysis difficult

### Cleaned Format
- **Shape**: 5,672 rows × 10 columns
- **Columns**:
  1. `Date` - Date of observation (2025-10-01 to 2025-10-31)
  2. `No` - Truck number
  3. `Jenis_DT` - Truck type (e.g., XCMG 400, Fuso Fighter)
  4. `No_Lambung` - Truck ID (e.g., DTXC 403)
  5. `Kategori` - Category (e.g., Coal Hauling)
  6. `Site` - Site location (e.g., Muara Enim)
  7. `Route` - Route name (**107-Port** or **BSR-107**)
  8. `Ritase` - Number of trips/cycles
  9. `Tonase` - Tonnage hauled
  10. `Keterangan` - Notes/remarks (shared across routes per date)

## Data Quality

### Distribution
- **Total records**: 5,672
- **107-Port route**: 2,971 records (52.4%)
- **BSR-107 route**: 2,701 records (47.6%)
- **Date range**: 31 days (Oct 1-31, 2025)

### Missing Values
- `Ritase`: 1,240 missing (21.9%)
- `Tonase`: 1,467 missing (25.9%)
- `Keterangan`: 4,097 missing (72.2%) - normal, as notes are optional
- Truck info (No, Jenis_DT, etc.): ~400 missing (7%) - likely blank rows from original

## Files

### Input
```
C:/Users/PC/Documents/1. Coal Hauling Project/2. Operations/Daily Report Dashboard/EN Daily Report.csv
```

### Output
```
C:/Users/PC/Documents/1. Coal Hauling Project/2. Operations/Daily Report Dashboard/EN Daily Report_CLEANED.csv
```

### Transformation Script
```
C:/Users/PC/PycharmProjects/DeepLearningFinance/ENDailyReport.py
```

## Usage Example

```python
import pandas as pd

# Load cleaned data
df = pd.read_csv("EN Daily Report_CLEANED.csv", parse_dates=['Date'])

# Analysis examples:
# 1. Total tonnage by route
df.groupby('Route')['Tonase'].sum()

# 2. Daily performance
df.groupby('Date')[['Ritase', 'Tonase']].sum()

# 3. Truck utilization
df.groupby('No_Lambung').agg({
    'Ritase': 'sum',
    'Tonase': 'sum',
    'Date': 'nunique'  # Days active
})

# 4. Route comparison by date
df.pivot_table(
    values='Tonase',
    index='Date',
    columns='Route',
    aggfunc='sum'
)
```

## Benefits

1. **Easier Analysis**: Standard pandas operations work seamlessly
2. **Better Visualization**: Can plot trends, comparisons, and distributions
3. **Scalable**: Easy to add new dates, routes, or trucks
4. **Standard Format**: Compatible with data science tools (scikit-learn, statsmodels, etc.)
5. **Maintainable**: Clear column meanings, no ambiguity

## Notes

- The `BA` route columns were present in the original data but were excluded from melting per requirements
- Empty rows and missing values were preserved to maintain data integrity
- Date assignment assumes each column group represents consecutive days starting Oct 1, 2025
