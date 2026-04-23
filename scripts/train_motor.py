"""
DevelAI Stage 1: Motor Learning Training
Train the motor system to perform object manipulation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from develai import ManipulationEnv, PhysicsObject, MotorLearningSystem
from tqdm import tqdm


def create_manipulation_task():
    """Create a simple manipulation task"""
    env = ManipulationEnv(width=100, height=100)
    
    # Add target object
    target = PhysicsObject(x=70, y=70, width=5, height=5, mass=1.0)
    env.add_object(target)
    env.set_target(target)
    
    return env


def train_motor_system(
    num_episodes: int = 1000,
    max_steps: int = 200,
    save_every: int = 100
):
    """
    Train the motor learning system
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_every: Save checkpoint every N episodes
    """
    # Create environment
    env = create_manipulation_task()
    
    # Get observation dimension
    obs = env._get_observation()
    state_dim = len(obs)
    action_dim = 3  # 3 arm segments
    
    # Create motor learning system
    motor_system = MotorLearningSystem(state_dim, action_dim, hidden_dim=256)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    grasps = []
    
    print(f"Starting motor learning training...")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps}")
    print()
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        env.reset()
        state = env._get_observation()
        episode_reward = 0
        episode_grasps = 0
        
        for step in range(max_steps):
            # Select action
            action, info = motor_system.select_action(state, deterministic=False)
            
            # Step environment
            result = env.step(action, dt=0.1)
            next_state = result['observation']
            reward = result['reward']
            done = result['done']
            grasped = result['info']['grasped']
            
            if grasped:
                episode_grasps += 1
            
            # Update motor system
            metrics = motor_system.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        grasps.append(episode_grasps)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_grasps = np.mean(grasps[-50:])
            print(f"Episode {episode + 1}: Avg Reward (last 50): {avg_reward:.2f}, Avg Grasps: {avg_grasps:.2f}")
        
        # Save checkpoint
        if (episode + 1) % save_every == 0:
            save_checkpoint(motor_system, episode + 1, episode_rewards, episode_lengths, grasps)
    
    # Final save
    save_checkpoint(motor_system, num_episodes, episode_rewards, episode_lengths, grasps)
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, grasps)
    
    return motor_system, episode_rewards, episode_lengths, grasps


def save_checkpoint(motor_system, episode, rewards, lengths, grasps):
    """Save training checkpoint"""
    checkpoint = {
        'episode': episode,
        'motor_cortex': motor_system.motor_cortex.state_dict(),
        'cerebellum': motor_system.cerebellum.state_dict(),
        'basal_ganglia': motor_system.basal_ganglia.state_dict(),
        'rewards': rewards,
        'lengths': lengths,
        'grasps': grasps
    }
    
    os.makedirs('checkpoints/develai', exist_ok=True)
    torch.save(checkpoint, f'checkpoints/develai/motor_checkpoint_ep{episode}.pt')
    print(f"Saved checkpoint at episode {episode}")


def plot_training_curves(rewards, lengths, grasps):
    """Plot training progress (save as text data)"""
    os.makedirs('results/develai', exist_ok=True)
    
    # Save training data as text
    with open('results/develai/training_data.txt', 'w') as f:
        f.write("Episode,Reward,Length,Grasps\n")
        for i, (r, l, g) in enumerate(zip(rewards, lengths, grasps)):
            f.write(f"{i},{r},{l},{g}\n")
    
    # Print summary statistics
    window = 50
    rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    grasps_smooth = np.convolve(grasps, np.ones(window)/window, mode='valid')
    
    print("\n=== Training Summary ===")
    print(f"Total episodes: {len(rewards)}")
    print(f"Final average reward (last 50): {rewards_smooth[-1]:.2f}")
    print(f"Final average grasps (last 50): {grasps_smooth[-1]:.2f}")
    print(f"Best reward: {max(rewards):.2f}")
    print(f"Best grasps in episode: {max(grasps)}")
    print("Saved training data to results/develai/training_data.txt")


def test_motor_system(motor_system, num_episodes: int = 10):
    """Test the trained motor system"""
    env = create_manipulation_task()
    motor_system.eval()
    
    print("\nTesting trained motor system...")
    
    for episode in range(num_episodes):
        env.reset()
        state = env._get_observation()
        episode_reward = 0
        episode_grasps = 0
        
        for step in range(200):
            action, info = motor_system.select_action(state, deterministic=True)
            result = env.step(action, dt=0.1)
            next_state = result['observation']
            reward = result['reward']
            grasped = result['info']['grasped']
            
            if grasped:
                episode_grasps += 1
            
            episode_reward += reward
            state = next_state
            
            if step % 50 == 0:
                print(f"  Step {step}: Reward={reward:.2f}, Grasped={grasped}")
        
        print(f"Episode {episode + 1}: Total Reward={episode_reward:.2f}, Grasps={episode_grasps}")
    
    motor_system.train()


if __name__ == "__main__":
    # Train motor system
    motor_system, rewards, lengths, grasps = train_motor_system(
        num_episodes=1000,
        max_steps=200,
        save_every=100
    )
    
    # Test trained system
    test_motor_system(motor_system, num_episodes=5)
    
    print("\nMotor learning training complete!")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Final average grasps (last 100 episodes): {np.mean(grasps[-100:]):.2f}")
