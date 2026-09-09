#!/usr/bin/env python3
"""Convenience entry point for running predictions.

Usage:
    python run_predict.py path/to/image.jpg
    python run_predict.py path/to/image_dir/ --model checkpoints/model_best.pth
    python run_predict.py --help
"""

from sign_language_cnn.predict import main

if __name__ == '__main__':
    main()
