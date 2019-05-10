import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        offset_reward = 1.
#         reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
#         reward = 1.-.005*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        current_position = self.sim.pose[0:3]
        current_euler_angles_sum = abs(self.sim.pose[4:6]).sum()
        
        #get the absolute distance between destination and current position
        distance_from_destination = np.sqrt(
            (current_position[0] - self.target_pos[0]) ** 2 + 
            (current_position[1] - self.target_pos[1]) ** 2 + 
            (current_position[2] - self.target_pos[2]) ** 2
        )
#         print(abs(current_euler_angles).sum())
        
        reward = np.tanh(offset_reward - 0.007*distance_from_destination) 
        if current_euler_angles_sum > 4.0:
            reward *= 0.95
        elif current_euler_angles_sum > 2.5:
            reward *= 0.97
        elif current_euler_angles_sum > 1.0:
            pass
        elif current_euler_angles_sum > 0.5:
            reward *= 1.01
        else:
            reward *= 1.02
            
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state