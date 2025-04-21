# constants.py

import os

# Environment name (already using Gymnasium naming)
ENV_NAME = 'ALE/Breakout-v5'  # This is correct for modern Gymnasium

# Number of parallel training threads
NUM_THREADS = 8

# Input frame dimensions (Atari standard preprocessed frame size)
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
STATE_LENGTH = 4  # Number of stacked frames

# Training parameters
GLOBAL_T_MAX = 100_000_000    # Total number of global steps
LOCAL_T_MAX = 5               # Max steps per thread before update
GAMMA = 0.99                  # Discount factor
ENTROPY_BETA = 0.01           # Entropy regularization constant

# Optimizer settings
INITIAL_LEARNING_RATE = 0.0007
RMSP_ALPHA = 0.99
RMSP_EPSILON = 1e-7  # Updated for TensorFlow 2.x recommended value

# Environment interaction
NO_OP_STEPS = 30              # Random no-ops at episode start

# Gradient clipping
CLIP_NORM = 40

# Checkpoint and logging
SAVE_INTERVAL = 500_000       # Save model every X global steps
LOG_INTERVAL = 10_000         # Log summary every X global steps

# Directory paths for saving
SAVE_NETWORK_PATH = os.path.join(os.getcwd(), 'saved_networks', ENV_NAME.replace('/', '_'))  # Fixed path issues
SAVE_SUMMARY_PATH = os.path.join(os.getcwd(), 'summary', ENV_NAME.replace('/', '_'))  # Fixed path issues

# Whether to load a previously saved network
LOAD_NETWORK = False

# Whether to render the environment for visualization
DISPLAY = False
FRAME_RATE = 60  # FPS for rendering

# Create directories if they don't exist
os.makedirs(SAVE_NETWORK_PATH, exist_ok=True)
os.makedirs(SAVE_SUMMARY_PATH, exist_ok=True)