#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import glob
import logging

# Set UTF-8 encoding for console output (Windows compatibility)
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append('.')
from Implementation.implementation import implementation

# Configure logging to only show INFO and above for cleaner output
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

def find_csv_file():
    """Automatically find CSV file in Implementation or other_data directories"""
    print('🔍 Searching for CSV files...')
    
    # First check Implementation directory
    impl_csvs = glob.glob('Implementation/*.csv')
    if impl_csvs:
        if len(impl_csvs) > 1:
            print(f'📁 Found {len(impl_csvs)} CSV files in Implementation directory:')
            for i, csv in enumerate(impl_csvs):
                print(f'   {i+1}. {csv}')
            print(f'✅ Using: {impl_csvs[0]}')
        else:
            print(f'✅ Found CSV file in Implementation directory: {impl_csvs[0]}')
        return impl_csvs[0]
    
    # If no CSV in Implementation, check other_data directory
    other_csvs = glob.glob('other_data/*.csv')
    if other_csvs:
        if len(other_csvs) > 1:
            print(f'📁 Found {len(other_csvs)} CSV files in other_data directory:')
            for i, csv in enumerate(other_csvs):
                print(f'   {i+1}. {csv}')
            print(f'✅ Using: {other_csvs[0]}')
        else:
            print(f'✅ Found CSV file in other_data directory: {other_csvs[0]}')
        return other_csvs[0]
    
    # No CSV files found
    raise FileNotFoundError("❌ No CSV files found in Implementation/ or other_data/ directories")

def run_implementation():
    print('🚀 Running Full Simulation with Enhanced Wide Report...')
    
    # Automatically find CSV file
    csv_file = find_csv_file()
    
    # Initialize implementation with SavedModel
    impl = implementation('models/saved_model', csv_file, 'config.ini')
    impl.load_dependencies()
    
    # Run simulation
    results = impl.run_simulation()
    
    if results:
        print('✅ Implementation completed successfully!')
        print('📊 Reports generated:')
        print('  - reports/performance_report_saved_model.csv (long format)')
        print('  - reports/performance_report_saved_model_wide.csv (wide format)')
        return True
    else:
        print('❌ Simulation failed')
        return False

if __name__ == "__main__":
    run_implementation()
