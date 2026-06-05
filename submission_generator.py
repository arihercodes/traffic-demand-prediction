import pandas as pd
import numpy as np

def create_submission(predictions, test_df, output_path='submission.csv'):
    submission = pd.DataFrame({
        'Index': test_df['Index'],
        'demand': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission.shape}")
    
    return submission

def validate_submission(submission_df, expected_rows=41778):

    print("\n" + "="*50)
    print("Validating Submission")
    print("="*50)
    
    # Check shape
    assert submission_df.shape[0] == expected_rows, \
        f"Expected {expected_rows} rows, got {submission_df.shape[0]}"
    assert submission_df.shape[1] == 2, \
        f"Expected 2 columns, got {submission_df.shape[1]}"
    
    # Check columns
    assert 'Index' in submission_df.columns, "Missing 'Index' column"
    assert 'demand' in submission_df.columns, "Missing 'demand' column"
    
    # Check value range
    assert submission_df['demand'].between(0, 1).all(), \
        "demand values should be between 0 and 1"
    
    # Check for missing values
    assert not submission_df.isnull().any().any(), "Found missing values in submission"
    
    print("All validation checks passed!")
    print(f"  - Rows: {submission_df.shape[0]}")
    print(f"  - Columns: {list(submission_df.columns)}")
    print(f"  - demand range: [{submission_df['demand'].min():.4f}, {submission_df['demand'].max():.4f}]")
    print(f"  - demand mean: {submission_df['demand'].mean():.4f}")
    print(f"  - demand std: {submission_df['demand'].std():.4f}")
    
    return True