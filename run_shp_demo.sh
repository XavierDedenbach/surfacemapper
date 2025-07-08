#!/bin/bash

echo "=========================================="
echo "SHP Workflow End-to-End Demo"
echo "=========================================="

# Stop existing containers
echo "Stopping existing containers..."
docker-compose down

# Rebuild containers
echo "Rebuilding containers..."
docker-compose build --no-cache

# Start containers
echo "Starting containers..."
docker-compose up -d

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
sleep 30

# Run the SHP workflow test
echo "Running SHP workflow test..."
python test_shp_workflow.py

# Show container logs if test fails
if [ $? -ne 0 ]; then
    echo "Test failed. Showing container logs..."
    docker-compose logs backend
fi

echo "Demo completed!" 