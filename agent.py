# coding:utf-8
# agent.py
import time
import random
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize

from network import A3CFF
from constant import ENV_NAME, FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH, DISPLAY
from constant import GLOBAL_T_MAX, LOCAL_T_MAX, GAMMA, INITIAL_LEARNING_RATE
from constant import NO_OP_STEPS, SAVE_INTERVAL, SAVE_NETWORK_PATH
from constant import LOG_INTERVAL, CLIP_NORM

class Agent:
    def __init__(self, thread_id, num_actions, global_network, optimizer):
        self.thread_id = thread_id
        self.local_network = A3CFF(num_actions)
        self.global_network = global_network
        self.optimizer = optimizer
        
        # Initialize local network with same architecture
        dummy_state = tf.random.normal([1, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH])
        self.local_network(dummy_state)  # Build the model by calling it once
        
        # Initial sync with global network
        self.sync_with_global()

    def sync_with_global(self):
        """Copy weights from global network to local network"""
        self.local_network.set_weights(self.global_network.get_weights())

    def get_initial_state(self, observation, last_observation):
        """Initialize state with stacked grayscale frames"""
        # Convert RGB to grayscale and resize
        processed = np.maximum(observation, last_observation)
        processed = rgb2gray(processed)  # Normalized between 0 and 1
        processed = resize(processed, (FRAME_WIDTH, FRAME_HEIGHT), anti_aliasing=True)
        state = [processed for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=2)

    def get_action(self, state):
        """Choose action using the local policy network"""
        state_tensor = tf.convert_to_tensor(state[np.newaxis, ...], dtype=tf.float32)
        pi, _ = self.local_network(state_tensor)
        pi = pi.numpy()[0]
        # Ensure valid probability distribution
        pi = pi / np.sum(pi)
        return np.random.choice(len(pi), p=pi)

    def preprocess(self, observation, last_observation):
        """Preprocess a single frame: max of two frames, grayscale, resize"""
        processed = np.maximum(observation, last_observation)
        processed = rgb2gray(processed)
        processed = resize(processed, (FRAME_WIDTH, FRAME_HEIGHT), anti_aliasing=True)
        return processed[..., np.newaxis]

    def compute_loss_and_apply_grads(self, state_batch, action_batch, reward_batch):
        """Compute loss and apply gradients to global network"""
        with tf.GradientTape() as tape:
            # Forward pass through the network
            state_tensor = tf.convert_to_tensor(state_batch, dtype=tf.float32)
            loss = self.local_network.compute_loss(state_tensor, action_batch, reward_batch)

        # Get gradients and apply them to global network
        grads = tape.gradient(loss, self.local_network.trainable_variables)
        clipped_grads = [tf.clip_by_norm(grad, CLIP_NORM) for grad in grads]
        
        # Apply gradients to global network
        self.optimizer.apply_gradients(zip(clipped_grads, self.global_network.trainable_variables))
        
        return loss.numpy()

    def save_network(self, ckpt_manager, global_t):
        """Save the global network checkpoint"""
        ckpt_manager.save(checkpoint_number=global_t)
        print(f'Successfully saved model at step {global_t}')

    def extract_scalar(self, value_output):
        """Extract scalar value from network output tensor"""
        if isinstance(value_output, (int, float, np.number)):
            return value_output
        elif isinstance(value_output, np.ndarray):
            if value_output.size == 1:
                return float(value_output.item())
            elif value_output.ndim == 1:
                return float(value_output[0])
        elif tf.is_tensor(value_output):
            return float(value_output.numpy())
        return 0.0  # Default return if extraction fails

    def actor_learner_thread(self, env, summary_writer):
        """Main training loop for each agent thread"""
        # Import pygame for event handling if this is the rendering thread
        # if self.thread_id == 0 and DISPLAY:
        #     import pygame
        
        global_t = 0
        local_t = 0
        learning_rate = INITIAL_LEARNING_RATE
        lr_step = INITIAL_LEARNING_RATE / GLOBAL_T_MAX

        total_reward, total_loss = 0, []
        duration = 0
        global_episode, local_episode = 0, 0
        pre_global_t_save, pre_global_t_log = 0, 0
        start_time = time.time()

        terminal = False
        observation, info = env.reset()  # Updated for Gymnasium API
        last_observation = observation
        
        # Perform random no-ops at start
        for _ in range(random.randint(1, NO_OP_STEPS)):
            last_observation = observation
            observation, _, terminal, truncated, _ = env.step(0)  # Do nothing
            
            # Process pygame events if this is the rendering thread
            if self.thread_id == 0 and DISPLAY:
                # pygame.event.pump()
                time.sleep(1/FRAME_RATE)  # Maintain frame rate during no-ops too
                
            if terminal or truncated:
                observation, info = env.reset()
                
        state = self.get_initial_state(observation, last_observation)

        # Setup checkpoint manager
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.global_network)
        ckpt_manager = tf.train.CheckpointManager(ckpt, SAVE_NETWORK_PATH, max_to_keep=3)

        try:
            while global_t < GLOBAL_T_MAX:
                local_t_start = local_t
                state_batch, action_batch, reward_batch = [], [], []

                # Sync local network with global network
                self.sync_with_global()

                while not (terminal or (local_t - local_t_start == LOCAL_T_MAX)):
                    last_observation = observation
                    action = self.get_action(state)
                    observation, reward, terminal, truncated, _ = env.step(action)  # Updated for Gymnasium API
                    
                    # Handle pygame events and add delay for the rendering thread
                    if self.thread_id == 0 and DISPLAY:
                        # pygame.event.pump()  # Process pygame events to keep the window responsive
                        time.sleep(1/FRAME_RATE)  # ~30 FPS rendering pace
                    
                    terminal = terminal or truncated  # Treat truncated as terminal for RL purposes

                    state_batch.append(state)
                    action_batch.append(action)
                    reward_batch.append(np.clip(reward, -1, 1))

                    processed = self.preprocess(observation, last_observation)
                    state = np.append(state[:, :, 1:], processed, axis=2)

                    total_reward += reward
                    local_t += 1
                    global_t += 1
                    duration += 1

                    learning_rate = max(0.0, learning_rate - lr_step)

                # Skip update if no experiences collected
                if len(reward_batch) == 0:
                    continue
                    
                # Bootstrap value for non-terminal states
                R = 0.0
                if not terminal:
                    state_tensor = tf.convert_to_tensor(state[np.newaxis, ...], dtype=tf.float32)
                    _, value = self.local_network(state_tensor)
                    R = self.extract_scalar(value)

                # Calculate discounted rewards
                discounted_rewards = np.zeros(len(reward_batch), dtype=np.float32)
                for i in reversed(range(len(reward_batch))):
                    R = reward_batch[i] + GAMMA * R
                    discounted_rewards[i] = R

                # Train network
                loss = self.compute_loss_and_apply_grads(
                    np.array(state_batch, dtype=np.float32),
                    np.array(action_batch, dtype=np.int32),
                    np.array(discounted_rewards, dtype=np.float32)
                )
                total_loss.append(loss)

                if terminal:
                    # Log episode statistics
                    with summary_writer.as_default():
                        tf.summary.scalar('total_reward', total_reward, step=global_episode + 1)
                        tf.summary.scalar('average_loss', np.mean(total_loss) if total_loss else 0, step=global_episode + 1)
                        tf.summary.scalar('episode_length', duration, step=global_episode + 1)

                    print(f'THREAD: {self.thread_id} / EPISODE: {global_episode + 1} / GLOBAL_T: {global_t} / '
                        f'TOTAL_REWARD: {total_reward:.2f} / AVG_LOSS: {np.mean(total_loss) if total_loss else 0:.5f}')

                    total_reward, total_loss, duration = 0, [], 0
                    local_episode += 1
                    global_episode += 1

                    terminal = False
                    observation, info = env.reset()  # Updated for Gymnasium API
                    last_observation = observation
                    
                    # Perform random no-ops at start
                    for _ in range(random.randint(1, NO_OP_STEPS)):
                        last_observation = observation
                        observation, _, terminal, truncated, _ = env.step(0)
                        
                        # Process pygame events during no-ops too
                        if self.thread_id == 0 and DISPLAY:
                            # pygame.event.pump()
                            time.sleep(1/FRAME_RATE)  # Maintain frame rate during no-ops
                            
                        if terminal or truncated:
                            observation, info = env.reset()
                            break
                    
                    state = self.get_initial_state(observation, last_observation)

                # Save network periodically (only thread 0)
                if (self.thread_id == 0) and (global_t - pre_global_t_save >= SAVE_INTERVAL):
                    pre_global_t_save = global_t
                    self.save_network(ckpt_manager, global_t)

                # Log performance metrics (only thread 0)
                if (self.thread_id == 0) and (global_t - pre_global_t_log >= LOG_INTERVAL):
                    pre_global_t_log = global_t
                    elapsed_time = time.time() - start_time
                    steps_per_sec = global_t / elapsed_time
                    print(f'##### PERFORMANCE: {global_t} steps in {elapsed_time:.0f} sec => '
                        f'{steps_per_sec:.0f} steps/sec, {steps_per_sec * 3600 / 1e6:.2f}M steps/hour')
        
        except Exception as e:
            print(f"Error in thread {self.thread_id}: {e}")
            import traceback
            traceback.print_exc()