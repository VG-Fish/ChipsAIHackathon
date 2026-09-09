#!/usr/bin/env python3
"""Convenience entry point for training the Sign Language CNN.

Usage:
    python run_training.py
    python run_training.py --start-resolution 512 --epochs-per-round 5
    python run_training.py --help
"""

from sign_language_cnn.train import main

if __name__ == '__main__':
    main()
