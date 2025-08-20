#!/bin/bash

# Run the first instance of the Python script
echo "Starting first run..."
python qrsac.py --config configs/qrsac-normal-iqn-neutral/donkeycar.yaml --gpu 0 --seed 0

# Check if the first command was successful
if [ $? -eq 0 ]; then
    echo "First run completed successfully. Waiting for 1 minute before the next run..."
    sleep 60
    
    # Run the second instance of the Python script
    echo "Starting second run..."
    python qrsac.py --config configs/qrsac-normal-iqn-neutral/donkeycar.yaml --gpu 0 --seed 0
    
    # Check if the second command was successful
    if [ $? -eq 0 ]; then
        echo "Second run completed successfully. Waiting for 1 minute before the next run..."
        sleep 60
        
        # Run the third instance of the Python script
        echo "Starting third run..."
        python qrsac.py --config configs/qrsac-normal-iqn-neutral/donkeycar.yaml --gpu 0 --seed 0
        
        # Check if the third command was successful
        if [ $? -eq 0 ]; then
            echo "Third run completed successfully."
        else
            echo "Error: Third run failed."
            exit 1
        fi
    else
        echo "Error: Second run failed."
        exit 1
    fi
else
    echo "Error: First run failed."
    exit 1
fi

echo "All runs completed or an error occurred."
