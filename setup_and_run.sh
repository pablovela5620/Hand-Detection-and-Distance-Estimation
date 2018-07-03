#!/usr/bin/env bash
conda env create -f environment.yml
source activate hand_pose
python hand_detection.py
