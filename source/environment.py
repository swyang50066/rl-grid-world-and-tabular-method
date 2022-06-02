from collections import deque

import numpy as np
from math import floor


# Neumann neighboring (<=> Moore neighborig; 8 neighbors)
X_DISPLACEMENTS = [-1, 1, 0, 0]
Y_DISPLACEMENTS = [0, 0, -1, 1]


class GridWorldEnv(object):
    """
    """
    def __init__(
        self, 
        map_size=(10, 10), 
        start_pos=(0, 0),
        end_pos=(9, 9),
        step_reward=1,
        positive_reward=10,
        negative_reward=-5,
        valid_transition_prob=0.7,
        transition_prob_bias=0.5
    ):
        # Parameters
        self.num_action = 4    # down/up/left/right
       
        self.grid_world = np.zeros(map_size) 
        self.map_size = map_size
        self.num_row, self.num_col = map_size
        self.num_state = self.num_row*self.num_col + 1

        self.start_pos = start_pos
        self.end_pos = end_pos
        self.grid_world[start_pos] = 1
        self.grid_world[end_pos] = 2
        
        self.start_state = self.to_state(start_pos)
        self.end_state = self.to_state(end_pos)
        self.states = list(range(self.num_row*self.num_col))
        
        self.step_reward = step_reward
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        
        self.valid_transition_prob = valid_transition_prob
        self.transition_prob_bias = transition_prob_bias

    def to_state(self, pos):
        return pos[0]*self.num_col + pos[1]
 
    def to_position(self, state):
        return (state // self.num_col, state % self.num_col)

    def build(self, epsilon=0.5):
        """Build random grid-world based on graph generation"""
        return self.grid_world

    '''
    def _get_direction(self, action, direction):
        """
        Takes is a direction and an action and returns a new direction.

        Parameters
        ----------
        action : int
            The current action 0, 1, 2, 3 for gridworld.

        direction : int
            Either -1, 0, 1.

        Returns
        -------
        direction : int
            Value either 0, 1, 2, 3.
        """
        left = [2,3,1,0]
        right = [3,2,0,1]
        if direction == 0:
            new_direction = action
        elif direction == -1:
            new_direction = left[action]
        elif direction == 1:
            new_direction = right[action]
        else:
            raise Exception("getDir received an unspecified case")
        return new_direction

    def _get_state(self, state, direction):
        """
        Get the next_state from the current state and a direction.

        Parameters
        ----------
        state : int
            The current state.

        direction : int
            The current direction.

        Returns
        -------
        next_state : int
            The next state given the current state and direction.
        """
        row_change = [-1,1,0,0]
        col_change = [0,0,-1,1]
        row_col = seq_to_col_row(state, self.num_cols)
        row_col[0,0] += row_change[direction]
        row_col[0,1] += col_change[direction]

        # check for invalid states
        if self.obs_states is not None:
            if (np.any(row_col < 0) or
                np.any(row_col[:,0] > self.num_rows-1) or
                np.any(row_col[:,1] > self.num_cols-1) or
                np.any(np.sum(abs(self.obs_states - row_col), 1)==0)):
                next_state = state
            else:
                next_state = row_col_to_seq(row_col, self.num_cols)[0]
        else:
            if (np.any(row_col < 0) or
                np.any(row_col[:,0] > self.num_rows-1) or
                np.any(row_col[:,1] > self.num_cols-1)):
                next_state = state
            else:
                next_state = row_col_to_seq(row_col, self.num_cols)[0]

        return next_state

    def set_reward_space(self):
        self.rewards = self.step_reward*np.ones((self.num_state,))
        self.rewards[-1] = 0
        self.rewards[self.end_state] = self.positive_reward

        for i in range(self.num_bad_states):
            if self.r_bad is None:
                raise Exception("Bad state specified but no reward is given")
            bad_state = row_col_to_seq(self.bad_states[i,:].reshape(1,-1), self.num_cols)
            self.R[bad_state, :] = self.r_bad
        for i in range(self.num_restart_states):
            if self.r_restart is None:
                raise Exception("Restart state specified but no reward is given")
            restart_state = row_col_to_seq(self.restart_states[i,:].reshape(1,-1), self.num_cols)
            self.R[restart_state, :] = self.r_restart
    
    def set_transition_prob(self):
        """Build the grid world"""
        self.P = np.zeros((self.num_states,self.num_states,self.num_actions))
        for action in range(self.num_actions):
            for state in range(self.num_states):

                # check if state is the fictional end state - self transition
                if state == self.num_states-1:
                    self.P[state, state, action] = 1
                    continue

                # check if the state is the goal state or an obstructed state - transition to end
                row_col = seq_to_col_row(state, self.num_cols)
                if self.obs_states is not None:
                    end_states = np.vstack((self.obs_states, self.goal_states))
                else:
                    end_states = self.goal_states

                if any(np.sum(np.abs(end_states-row_col), 1) == 0):
                    self.P[state, self.num_states-1, action] = 1

                # else consider stochastic effects of action
                else:
                    for dir in range(-1,2,1):
                        direction = self._get_direction(action, dir)
                        next_state = self._get_state(state, direction)
                        if dir == 0:
                            prob = self.p_good_trans
                        elif dir == -1:
                            prob = (1 - self.p_good_trans)*(self.bias)
                        elif dir == 1:
                            prob = (1 - self.p_good_trans)*(1-self.bias)

                        self.P[state, next_state, action] += prob

                # make restart states transition back to the start state with
                # probability 1
                if self.restart_states is not None:
                    if any(np.sum(np.abs(self.restart_states-row_col),1)==0):
                        next_state = row_col_to_seq(self.start_state, self.num_cols)
                        self.P[state,:,:] = 0
                        self.P[state,next_state,:] = 1
        return self
    '''

env = GridWorldEnv(
    map_size=(8, 8),
    start_pos=(0, 0),
    end_pos=(7, 7),
    step_reward=1,
    positive_reward=10,
    negative_reward=-5,
    valid_transition_prob=0.7,
    transition_prob_bias=0.5
)

grid_world = env.build(num_obstacle=10)
print("@"*20)
print(grid_world)
