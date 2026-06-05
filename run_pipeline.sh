echo "Traffic Demand Prediction Pipeline"


# Check if input files exist
echo ""
echo "Checking input files..."

if [ ! -f "train.csv" ]; then
    echo "ERROR: train.csv not found!"
    exit 1
fi

if [ ! -f "test.csv" ]; then
    echo "ERROR: test.csv not found!"
    exit 1
fi

echo "Input files found"

# Install dependencies (uncomment if needed)
# echo ""
# echo "Installing dependencies..."
# pip install -r requirements.txt

# Run the main script
echo "Running prediction pipeline..."
python traffic_demand_prediction.py

# Check if submission was created
echo "Checking output..."

if [ -f "submission.csv" ]; then
    echo "submission.csv created successfully"
    
    # Show file info
    echo ""
    echo "Submission file info:"
    wc -l submission.csv
    head -5 submission.csv
    echo ""
    echo "Pipeline completed successfully!"
else
    echo "ERROR: submission.csv was not created!"
    exit 1
fi

echo "Done! Upload submission.csv to the platform"
