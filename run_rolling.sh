#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shane/Cocoa/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/shane/Cocoa/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/shane/Cocoa/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
/home/shane/Cocoa/.venv/bin/python /home/shane/Cocoa/src/cocoa/experiments/rolling_wll.py
