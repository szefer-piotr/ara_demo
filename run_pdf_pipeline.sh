#!/bin/bash

# Set the PYTHONPATH to include the project root
export PYTHONPATH="/home/piotr/projects/ara_demo/ara_demo:$PYTHONPATH"

# Run the PDF processing pipeline
python3 /home/piotr/projects/ara_demo/ara_demo/src/pdf_processing_pipeline.py "$@"

