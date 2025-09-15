#!/usr/bin/env python3
"""
Dataset Preprocessing and Merging Script

Preprocesses and merges all datasets in the data folder into a single, 
training-ready dataset with consistent schema and cleaned data.

Usage:
    python preprocess_and_merge.py --output merged_training_data.csv
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define the data directory
DATA_DIR = Path(__file__).parent / 'data'

def standardize_column_names(df):
    """Standardize column names across datasets."""
    # Define column mapping for standardization
    column_mapping = {
        'precision_val': 'precision',
        'data_source': 'source_type'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    return df

def add_missing_columns(df, required_columns):
    """Add missing columns with appropriate default values."""
    for col in required_columns:
        if col not in df.columns:
            if col == 'model_name':
                df[col] = 'synthetic_model'
            elif col == 'run_id':
                # For synthetic data, create sequential run IDs
                df[col] = range(1, len(df) + 1)
            elif col == 'source_type':
                df[col] = 'unknown'
            else:
                df[col] = 0  # Default numeric value
    return df

def clean_numeric_columns(df):
    """Clean and validate numeric columns."""
    numeric_columns = [
        'total_flops', 'total_params', 'array_size', 'precision', 
        'batch_size', 'clock_ghz', 'latency_ms', 'power_w', 
        'memory_mb', 'throughput_ops_s', 'run_id'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, replacing any non-numeric values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle specific column constraints
            if col in ['total_flops', 'total_params', 'throughput_ops_s']:
                # These should be positive
                df[col] = df[col].abs()
            elif col == 'latency_ms':
                # Latency should be positive and reasonable (< 10 seconds)
                df[col] = df[col].abs()
                df.loc[df[col] > 10000, col] = np.nan  # Cap at 10 seconds
            elif col in ['power_w', 'memory_mb']:
                # These should be positive
                df[col] = df[col].abs()
            elif col == 'array_size':
                # Array size should be reasonable power of 2 or common values
                valid_sizes = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
                df.loc[~df[col].isin(valid_sizes), col] = 32  # Default to 32
            elif col == 'precision':
                # Precision should be 8, 16, or 32
                valid_precisions = [8, 16, 32]
                df.loc[~df[col].isin(valid_precisions), col] = 16  # Default to FP16
            elif col == 'batch_size':
                # Batch size should be positive integer
                df[col] = df[col].abs().round().astype(int)
                df.loc[df[col] == 0, col] = 1  # Minimum batch size of 1
    
    return df

def handle_outliers(df):
    """Handle outliers using IQR method for key performance metrics."""
    outlier_columns = ['latency_ms', 'power_w', 'memory_mb', 'throughput_ops_s']
    
    for col in outlier_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them (to preserve data)
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
    
    return df

def recalculate_throughput(df):
    """Recalculate throughput to ensure consistency."""
    if 'total_flops' in df.columns and 'latency_ms' in df.columns:
        # Throughput = FLOPs / (latency in seconds)
        df['throughput_ops_s'] = df['total_flops'] / (df['latency_ms'] / 1000)
        
        # Handle infinite or very large values
        df.loc[df['throughput_ops_s'] == np.inf, 'throughput_ops_s'] = np.nan
        df.loc[df['throughput_ops_s'] > 1e15, 'throughput_ops_s'] = np.nan
    
    return df

def add_engineered_features(df):
    """Add engineered features that might be useful for training."""
    
    # FLOPs per parameter ratio
    if 'total_flops' in df.columns and 'total_params' in df.columns:
        df['flops_per_param'] = df['total_flops'] / (df['total_params'] + 1e-8)
    
    # Compute intensity (FLOPs per byte assuming 4 bytes per param for FP32)
    if 'total_flops' in df.columns and 'total_params' in df.columns:
        bytes_per_param = df['precision'] / 8  # Convert bits to bytes
        total_bytes = df['total_params'] * bytes_per_param
        df['compute_intensity'] = df['total_flops'] / (total_bytes + 1e-8)
    
    # Array utilization (total params vs array capacity)
    if 'total_params' in df.columns and 'array_size' in df.columns:
        array_capacity = df['array_size'] ** 2  # Assuming square array
        df['array_utilization'] = np.minimum(df['total_params'] / (array_capacity * 1000), 1.0)
    
    # Power efficiency (FLOPs per Watt)
    if 'total_flops' in df.columns and 'power_w' in df.columns:
        df['power_efficiency'] = df['total_flops'] / (df['power_w'] + 1e-8)
    
    # Memory efficiency (FLOPs per MB)
    if 'total_flops' in df.columns and 'memory_mb' in df.columns:
        df['memory_efficiency'] = df['total_flops'] / (df['memory_mb'] + 1e-8)
    
    return df

def load_and_preprocess_dataset(file_path):
    """Load and preprocess a single dataset file."""
    print(f"Processing: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Standardize column names
        df = standardize_column_names(df)
        
        # Define required columns for final dataset
        required_columns = [
            'model_name', 'total_flops', 'total_params', 'run_id',
            'array_size', 'precision', 'batch_size', 'clock_ghz',
            'latency_ms', 'power_w', 'memory_mb', 'source_type', 'throughput_ops_s'
        ]
        
        # Add missing columns
        df = add_missing_columns(df, required_columns)
        
        # Clean numeric columns
        df = clean_numeric_columns(df)
        
        # Handle outliers
        df = handle_outliers(df)
        
        # Recalculate throughput for consistency
        df = recalculate_throughput(df)
        
        # Add engineered features
        df = add_engineered_features(df)
        
        print(f"  After preprocessing: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return None

def merge_datasets(data_dir, output_file):
    """Merge all datasets in the data directory."""
    
    # Find all CSV files in the data directory
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print("=" * 60)
    
    all_dataframes = []
    
    # Process each file
    for file_path in csv_files:
        df = load_and_preprocess_dataset(file_path)
        if df is not None and len(df) > 0:
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("No valid datasets to merge!")
        return
    
    # Merge all dataframes
    print("\nMerging datasets...")
    merged_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    # Final cleanup
    print("Performing final cleanup...")
    
    # Remove rows with critical missing values
    critical_columns = ['total_flops', 'total_params', 'latency_ms']
    initial_rows = len(merged_df)
    merged_df = merged_df.dropna(subset=critical_columns)
    dropped_rows = initial_rows - len(merged_df)
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} rows with missing critical values")
    
    # Fill remaining NaN values with medians
    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if merged_df[col].isna().any():
            median_val = merged_df[col].median()
            merged_df[col].fillna(median_val, inplace=True)
    
    # Final column ordering for training
    final_columns = [
        'model_name', 'source_type', 'run_id',
        'total_flops', 'total_params', 'flops_per_param',
        'array_size', 'precision', 'batch_size', 'clock_ghz',
        'compute_intensity', 'array_utilization',
        'latency_ms', 'power_w', 'memory_mb', 'throughput_ops_s',
        'power_efficiency', 'memory_efficiency'
    ]
    
    # Select and reorder columns
    available_columns = [col for col in final_columns if col in merged_df.columns]
    merged_df = merged_df[available_columns]
    
    # Save the merged dataset
    output_path = data_dir / output_file
    merged_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print(f"SUCCESS: Merged dataset saved to {output_path}")
    print(f"Final dataset shape: {merged_df.shape}")
    print(f"\nDataset composition:")
    print(merged_df['source_type'].value_counts())
    
    print(f"\nSummary statistics:")
    print(f"  Total models: {merged_df['model_name'].nunique()}")
    print(f"  Total runs: {merged_df['run_id'].nunique()}")
    print(f"  Latency range: {merged_df['latency_ms'].min():.2f} - {merged_df['latency_ms'].max():.2f} ms")
    print(f"  Power range: {merged_df['power_w'].min():.2f} - {merged_df['power_w'].max():.2f} W")
    print(f"  Memory range: {merged_df['memory_mb'].min():.2f} - {merged_df['memory_mb'].max():.2f} MB")
    
    print(f"\nColumns in final dataset: {list(merged_df.columns)}")
    
    return merged_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess and merge all datasets")
    parser.add_argument("--output", type=str, default="merged_training_data.csv",
                       help="Output filename for merged dataset")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Data directory (default: data/ subdirectory)")
    
    args = parser.parse_args()
    
    # Use provided data directory or default
    data_directory = Path(args.data_dir) if args.data_dir else DATA_DIR
    
    if not data_directory.exists():
        print(f"Data directory not found: {data_directory}")
        exit(1)
    
    # Process and merge datasets
    merged_df = merge_datasets(data_directory, args.output)