"""
Main entry point for Trash Classification project
"""

import argparse
import sys
from pathlib import Path

from src.config import Config
from src.train import train
from src.test import evaluate


def main():
    """
    Main function to run train or test
    """
    parser = argparse.ArgumentParser(description='Trash Classification using ConvNeXt Tiny')
    parser.add_argument(
        'mode',
        type=str,
        choices=['train', 'test', 'predict'],
        help='Mode to run: train, test, or predict'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=f'{Config.CHECKPOINT_DIR}/best_model.pth',
        help='Path to model checkpoint (for testing or resuming training)'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file or directory (for predict mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("="*60)
        print("TRAINING MODE")
        print("="*60)
        print(f"Model: ConvNeXt Tiny")
        print(f"Classes: {Config.CLASS_NAMES}")
        print(f"Device: {Config.DEVICE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print("="*60 + "\n")
        
        # train(checkpoint_path=args.checkpoint)
        train()
    
    elif args.mode == 'test':
        print("="*60)
        print("TESTING MODE")
        print("="*60)
        print(f"Checkpoint: {args.checkpoint}")
        print("="*60 + "\n")
        
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint file not found: {args.checkpoint}")
            print("Please train the model first using: python main.py train")
            sys.exit(1)
        
        evaluate(args.checkpoint)
    
    elif args.mode == 'predict':
        print("="*60)
        print("PREDICTION MODE")
        print("="*60)
        
        if not args.image:
            print("Error: --image argument is required for predict mode")
            print("Usage: python main.py predict --image <path_to_image>")
            sys.exit(1)
        
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint file not found: {args.checkpoint}")
            print("Please train the model first using: python main.py train")
            sys.exit(1)
        
        # Import and run predict
        from src.predict import main as predict_main
        sys.argv = ['predict', '--image', args.image, '--checkpoint', args.checkpoint]
        predict_main()


if __name__ == "__main__":
    main()
