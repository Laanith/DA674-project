# coding:utf-8
# main.py

import os
import sys
import time
import threading
import gymnasium as gym
import tensorflow as tf
from threading import Thread

from network import A3CFF
from agent import Agent
from constant import (
    ENV_NAME, NUM_THREADS, RMSP_ALPHA, RMSP_EPSILON, 
    SAVE_NETWORK_PATH, SAVE_SUMMARY_PATH, LOAD_NETWORK, 
    DISPLAY, INITIAL_LEARNING_RATE, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH, FRAME_RATE
)


def setup_environment():
    """Verify and setup the environment"""
    try:
        # Check if the environment is available
        gym.make(ENV_NAME)
        print(f"Successfully verified environment: {ENV_NAME}")
    except gym.error.NameNotRegisteredError:
        print(f"Error: Environment {ENV_NAME} not found. You may need to install atari-py or ROMs.")
        print("Try: pip install gymnasium[atari,accept-rom-license]")
        sys.exit(1)
    
    # Ensure directories exist
    os.makedirs(SAVE_NETWORK_PATH, exist_ok=True)
    os.makedirs(SAVE_SUMMARY_PATH, exist_ok=True)


def main():
    """Main function to run the A3C algorithm"""
    # Setup before starting
    setup_environment()
    
    # TensorFlow settings for performance
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"GPU acceleration enabled. Found {len(physical_devices)} GPU(s).")
        except Exception as e:
            print(f"Error setting GPU memory growth: {e}")
    else:
        print("No GPU found. Running on CPU.")
    
    # Create parallel environments
    envs = []
    for i in range(NUM_THREADS):
        render_mode = 'human' if (i == 0 and DISPLAY) else None
        env = gym.make(ENV_NAME, render_mode=render_mode)
        envs.append(env)
    
    # Setup FPS for rendering
    if DISPLAY:
        try:
            envs[0].metadata['render_fps'] = FRAME_RATE
        except:
            print("Could not set render_fps, continuing without setting it.")
    
    # Get number of actions for the environment
    num_actions = envs[0].action_space.n
    print(f"Environment: {ENV_NAME}")
    print(f"Action Space: {num_actions} discrete actions")
    print(f"Observation Shape: {envs[0].observation_space.shape}")
    
    # Initialize global network and optimizer
    global_network = A3CFF(num_actions)
    
    # Initialize weights with a dummy forward pass
    dummy_state = tf.zeros([1, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH])
    _, _ = global_network(dummy_state)
    
    # Create optimizer
    optimizer = tf.keras.optimizers.legacy.RMSprop(
        learning_rate=INITIAL_LEARNING_RATE, 
        rho=RMSP_ALPHA, 
        epsilon=RMSP_EPSILON
    )
    
    global_network.summary()
    
    # TensorBoard summary writer
    summary_writer = tf.summary.create_file_writer(SAVE_SUMMARY_PATH)
    
    # Load network checkpoint if available
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=global_network)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, SAVE_NETWORK_PATH, max_to_keep=3)
    
    if LOAD_NETWORK:
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint)
            print(f"Restored from {ckpt_manager.latest_checkpoint}")
        else:
            print("No checkpoint found. Training new network...")
    
    # Create agents
    agents = []
    for i in range(NUM_THREADS):
        agent = Agent(thread_id=i, num_actions=num_actions, global_network=global_network, optimizer=optimizer)
        agents.append(agent)
    
    # Create threads
    actor_learner_threads = []
    for i in range(NUM_THREADS):
        agent = agents[i]
        env = envs[i]
        thread = Thread(target=agent.actor_learner_thread, args=(env, summary_writer))
        thread.daemon = True  # Set as daemon so main process can exit
        actor_learner_threads.append(thread)
    
    # Start threads with small delay between them
    for i, thread in enumerate(actor_learner_threads):
        thread.start()
        print(f"Started thread {i}")
        time.sleep(1.0)  # Increased delay for better stability
    
    try:
        # Keep main thread alive while training
        print("Main thread monitoring training...")
        while any(thread.is_alive() for thread in actor_learner_threads):
            alive_threads = sum(1 for thread in actor_learner_threads if thread.is_alive())
            print(f"Active training threads: {alive_threads}/{NUM_THREADS}")
            time.sleep(60.0)  # Status update every minute
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Closing environments...")
    
    finally:
        # Close environments
        for env in envs:
            try:
                env.close()
            except:
                pass
        
        print("Training finished or interrupted. Environments closed.")


if __name__ == '__main__':
    main()