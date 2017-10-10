"""project settings."""
import os
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
if __name__ == '__main__':
    print(PROJECT_DIR)
