"""
DevelAI Simulation Environment
Simple 2D physics environment for motor learning
"""

import numpy as np
import torch
from typing import Tuple, Dict, List
import math


class PhysicsObject:
    """Simple 2D physics object"""
    
    def __init__(self, x: float, y: float, width: float, height: float, mass: float = 1.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.mass = mass
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0
        self.angular_velocity = 0.0
        
    def update(self, dt: float):
        """Update position based on velocity"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.angular_velocity * dt
        
        # Simple friction
        self.vx *= 0.99
        self.vy *= 0.99
        self.angular_velocity *= 0.99
        
    def get_center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class ArmSegment:
    """Single segment of a robotic arm"""
    
    def __init__(self, length: float, angle: float = 0.0):
        self.length = length
        self.angle = angle
        self.angular_velocity = 0.0
        self.torque = 0.0
        
    def update(self, dt: float):
        """Update angle based on torque"""
        # Simple angular acceleration
        angular_acceleration = self.torque / (self.length ** 2)
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt
        
        # Damping
        self.angular_velocity *= 0.95


class RoboticArm:
    """Multi-segment robotic arm for manipulation"""
    
    def __init__(self, base_x: float, base_y: float, segment_lengths: List[float]):
        self.base_x = base_x
        self.base_y = base_y
        self.segments = [ArmSegment(length) for length in segment_lengths]
        self.end_effector_pos = (base_x, base_y)
        
    def apply_torques(self, torques: List[float]):
        """Apply torques to each segment"""
        for i, torque in enumerate(torques):
            if i < len(self.segments):
                self.segments[i].torque = torque
                
    def update(self, dt: float):
        """Update arm kinematics"""
        x, y = self.base_x, self.base_y
        
        for segment in self.segments:
            segment.update(dt)
            x += segment.length * math.cos(segment.angle)
            y += segment.length * math.sin(segment.angle)
            
        self.end_effector_pos = (x, y)
        
    def get_joint_positions(self) -> List[Tuple[float, float]]:
        """Get positions of all joints"""
        positions = [(self.base_x, self.base_y)]
        x, y = self.base_x, self.base_y
        
        for segment in self.segments:
            x += segment.length * math.cos(segment.angle)
            y += segment.length * math.sin(segment.angle)
            positions.append((x, y))
            
        return positions
    
    def get_end_effector(self) -> Tuple[float, float]:
        return self.end_effector_pos


class ManipulationEnv:
    """Environment for object manipulation tasks"""
    
    def __init__(self, width: float = 100.0, height: float = 100.0):
        self.width = width
        self.height = height
        self.arm = RoboticArm(width / 2, height / 2, [15.0, 12.0, 8.0])
        self.objects = []
        self.target_object = None
        self.time = 0.0
        
    def add_object(self, obj: PhysicsObject):
        self.objects.append(obj)
        
    def set_target(self, obj: PhysicsObject):
        self.target_object = obj
        
    def reset(self):
        """Reset environment to initial state"""
        self.arm = RoboticArm(self.width / 2, self.height / 2, [15.0, 12.0, 8.0])
        self.time = 0.0
        
    def step(self, torques: List[float], dt: float = 0.1) -> Dict:
        """Step the simulation"""
        self.time += dt
        
        # Update arm
        self.arm.apply_torques(torques)
        self.arm.update(dt)
        
        # Update objects
        for obj in self.objects:
            obj.update(dt)
            
        # Check collisions (simple AABB)
        end_effector = self.arm.get_end_effector()
        grasped = False
        
        if self.target_object:
            obj_bounds = self.target_object.get_bounds()
            # Check if end effector is near object
            if (obj_bounds[0] <= end_effector[0] <= obj_bounds[2] and
                obj_bounds[1] <= end_effector[1] <= obj_bounds[3]):
                grasped = True
                
        # Calculate reward
        reward = self._calculate_reward(end_effector, grasped)
        
        return {
            'observation': self._get_observation(),
            'reward': reward,
            'done': False,
            'info': {'grasped': grasped, 'end_effector': end_effector}
        }
        
    def _calculate_reward(self, end_effector: Tuple[float, float], grasped: bool) -> float:
        """Calculate reward based on task completion"""
        if not self.target_object:
            return 0.0
            
        target_center = self.target_object.get_center()
        distance = math.sqrt(
            (end_effector[0] - target_center[0]) ** 2 +
            (end_effector[1] - target_center[1]) ** 2
        )
        
        # Reward for getting closer
        reward = -distance / 100.0
        
        # Bonus for grasping
        if grasped:
            reward += 10.0
            
        return reward
        
    def _get_observation(self) -> np.ndarray:
        """Get observation vector"""
        obs = []
        
        # Arm joint angles
        for segment in self.arm.segments:
            obs.extend([segment.angle, segment.angular_velocity])
            
        # End effector position
        end_effector = self.arm.get_end_effector()
        obs.extend(end_effector)
        
        # Target object position (if exists)
        if self.target_object:
            target_center = self.target_object.get_center()
            obs.extend(target_center)
            
        return np.array(obs, dtype=np.float32)
        
    def render(self):
        """Render the environment (ASCII for now)"""
        # Simple ASCII rendering
        grid = [['.' for _ in range(int(self.width))] for _ in range(int(self.height))]
        
        # Draw arm
        joints = self.arm.get_joint_positions()
        for i, (x, y) in enumerate(joints):
            if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                grid[int(y)][int(x)] = 'J' if i > 0 else 'B'
                
        # Draw target object
        if self.target_object:
            x, y = self.target_object.get_center()
            if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                grid[int(y)][int(x)] = 'T'
                
        # Print grid
        for row in reversed(grid):
            print(''.join(row))
        print()
