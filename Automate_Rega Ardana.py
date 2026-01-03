import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
import os
import sys
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION - Edit these settings
# ============================================================================
CONFIG = {
    'filepath': r'D:\Asah Academy\Membuat Sistem Machine Learning\FloodDataset_Raw\flood.csv',
    'target_column': 'FloodProbability',
    'test_size': 0.2,
    'random_state': 42,
    'scaling_method': 'standard',  # 'standard' or 'minmax'
    'missing_strategy': 'mean',     # 'mean', 'median', 'most_frequent'
    'save_processed': True,         # Save processed data to CSV
    'output_folder': 'Flood_Preprocessing'
}


# ============================================================================
# AUTOMATED PREPROCESSING FUNCTIONS
# ============================================================================

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_step(step_num, text):
    """Print step information."""
    print(f"\n[STEP {step_num}] {text}")


def load_data_auto(filepath):
    """Automatically load data from filepath."""
    print_step(1, "LOADING DATA")
    try:
        df = pd.read_csv(filepath)
        print(f" Data loaded successfully!")
        print(f" Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f" ERROR: File not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f" ERROR: {str(e)}")
        sys.exit(1)


def check_quality_auto(df):
    """Automatically check data quality."""
    print_step(2, "CHECKING DATA QUALITY")
    
    # Check missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    if total_missing > 0:
        print(f" Missing values detected: {total_missing} total")
        missing_cols = missing[missing > 0]
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            print(f"  - {col}: {count} ({percentage:.2f}%)")
    else:
        print(" No missing values found")
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f" Duplicate rows detected: {duplicates} ({(duplicates/len(df)*100):.2f}%)")
    else:
        print(" No duplicate rows found")
    
    # Check data types
    print(f"\nData types:")
    print(f"  - Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  - Non-numeric columns: {len(df.select_dtypes(exclude=[np.number]).columns)}")
    
    return total_missing, duplicates


def handle_missing_auto(df, strategy='mean'):
    """Automatically handle missing values."""
    if df.isnull().sum().sum() == 0:
        print_step(3, "HANDLING MISSING VALUES")
        print(" No missing values to handle - skipping")
        return df
    
    print_step(3, "HANDLING MISSING VALUES")
    print(f" Using '{strategy}' imputation strategy...")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Impute numeric columns
    if numeric_cols:
        imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print(f" Missing values imputed for {len(numeric_cols)} numeric columns")
    
    # Verify no missing values remain
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing == 0:
        print(" All missing values handled successfully")
    else:
        print(f" Warning: {remaining_missing} missing values remain")
    
    return df


def remove_duplicates_auto(df):
    """Automatically remove duplicate rows."""
    print_step(4, "REMOVING DUPLICATES")
    
    initial_rows = df.shape[0]
    df_cleaned = df.drop_duplicates()
    removed = initial_rows - df_cleaned.shape[0]
    
    if removed > 0:
        print(f" Removed {removed} duplicate rows ({(removed/initial_rows*100):.2f}%)")
        print(f" Remaining: {df_cleaned.shape[0]} rows")
    else:
        print(" No duplicates found - skipping")
    
    return df_cleaned.reset_index(drop=True)


def split_features_target_auto(df, target_column):
    """Automatically split features and target."""
    print_step(5, "SPLITTING FEATURES AND TARGET")
    
    if target_column not in df.columns:
        print(f" ERROR: Target column '{target_column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f" Features (X): {X.shape[1]} columns, {X.shape[0]} rows")
    print(f" Target (y): {y.name}")
    print(f" Feature columns: {list(X.columns)}")
    
    return X, y


def train_test_split_auto(X, y, test_size, random_state):
    """Automatically split train and test sets."""
    print_step(6, "SPLITTING TRAIN/TEST SETS")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    train_pct = int((1-test_size)*100)
    test_pct = int(test_size*100)
    
    print(f" Split ratio: {train_pct}% train / {test_pct}% test")
    print(f" Training set: {X_train.shape[0]} samples")
    print(f" Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features_auto(X_train, X_test, method='standard'):
    """Automatically scale features."""
    print_step(7, "SCALING FEATURES")
    
    # Initialize scaler
    if method == 'standard':
        scaler = StandardScaler()
        print(" Using StandardScaler (mean=0, std=1)")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        print(" Using MinMaxScaler (range 0-1)")
    else:
        print(f" ERROR: Unknown scaling method '{method}'")
        sys.exit(1)
    
    # Fit on training data and transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(" Features scaled successfully")
    print(f" Train set scaled: {X_train_scaled.shape}")
    print(f" Test set scaled: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def save_processed_data(X_train, X_test, y_train, y_test, output_folder):
    """Save processed data to CSV files."""
    print_step(8, "SAVING PROCESSED DATA")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f" Created folder: {output_folder}")
    
    # Save files
    try:
        X_train.to_csv(f'{output_folder}/X_train.csv', index=False)
        X_test.to_csv(f'{output_folder}/X_test.csv', index=False)
        y_train.to_csv(f'{output_folder}/y_train.csv', index=False, header=True)
        y_test.to_csv(f'{output_folder}/y_test.csv', index=False, header=True)
        
        print(" All processed data saved:")
        print(f"  - {output_folder}/X_train.csv")
        print(f"  - {output_folder}/X_test.csv")
        print(f"  - {output_folder}/y_train.csv")
        print(f"  - {output_folder}/y_test.csv")
    except Exception as e:
        print(f" Warning: Could not save files - {str(e)}")


def display_summary(X_train, X_test, y_train, y_test):
    """Display final summary."""
    print_header("PREPROCESSING COMPLETED SUCCESSFULLY!")
    
    print("\n FINAL DATA SUMMARY:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"  Target variable: {y_train.name}")
    
    print("\n DATA STATISTICS:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    print("\n Data is ready for model training!")
    print("="*70 + "\n")


# ============================================================================
# MAIN AUTOMATED EXECUTION
# ============================================================================

def run_automated_preprocessing():
    """Run the complete automated preprocessing pipeline."""
    
    print_header("AUTOMATED FLOOD PREDICTION DATA PREPROCESSING")
    print(f"Configuration loaded from CONFIG dictionary")
    print(f"Target file: {CONFIG['filepath']}")
    
    # Step 1: Load data
    df = load_data_auto(CONFIG['filepath'])
    
    # Step 2: Check data quality
    total_missing, duplicates = check_quality_auto(df)
    
    # Step 3: Handle missing values
    df = handle_missing_auto(df, strategy=CONFIG['missing_strategy'])
    
    # Step 4: Remove duplicates
    df = remove_duplicates_auto(df)
    
    # Step 5: Split features and target
    X, y = split_features_target_auto(df, CONFIG['target_column'])
    
    # Step 6: Train/test split
    X_train, X_test, y_train, y_test = train_test_split_auto(
        X, y, 
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    # Step 7: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features_auto(
        X_train, X_test,
        method=CONFIG['scaling_method']
    )
    
    # Step 8: Save processed data
    if CONFIG['save_processed']:
        save_processed_data(
            X_train_scaled, X_test_scaled, 
            y_train, y_test,
            CONFIG['output_folder']
        )
    
    # Display summary
    display_summary(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Return processed data
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X_train_scaled.columns)
    }


# ============================================================================
# AUTO-EXECUTION WHEN SCRIPT IS RUN
# ============================================================================

if __name__ == "__main__":
    # This will automatically run when you execute: python flood_preprocessing.py
    processed_data = run_automated_preprocessing()
    
    # Optional: Print first few rows of processed data
    print("\n PREVIEW OF PROCESSED DATA:")
    print("\nX_train (first 5 rows):")
    print(processed_data['X_train'].head())
    print("\ny_train (first 5 values):")
    print(processed_data['y_train'].head())