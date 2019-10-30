#!/usr/bin/env bash
set -e

echo 'running splitter...'
python3 src/preprocess/splitter.py

echo 'running baseline_algorithm...'
python3 src/baseline_algorithm/rec_popular.py
echo 'done'

echo 'running verify_subm...'
python3 src/verify_submission/verify_subm.py
echo 'done'

echo 'running score_subm...'
python3 src/score_submission/score_subm.py
echo 'done'
