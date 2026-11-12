import numpy as np

ACTIONS = np.array(["U", "D", "L", "R"])
ARROWS = np.array(["↑", "↓", "←", "→"])


class MDP:
    """Finite MDP with transition array T[a, s, s'] and reward array R[a, s]."""

    def __init__(self, T, R, discount):
        assert T.ndim == 3 and T.shape[1] == T.shape[2], "T must have shape (nActions, nStates, nStates)"
        assert np.allclose(T.sum(axis=2), 1), "Each row of T must sum to 1"
        assert R.shape == T.shape[:2], "R must have shape (nActions, nStates)"
        assert 0 <= discount < 1, "discount must be in [0, 1)"
        self.T, self.R, self.discount = T, R, discount
        self.nActions, self.nStates = T.shape[:2]

    def q_values(self, V):
        return self.R + self.discount * self.T @ V

    def extract_policy(self, V):
        return np.argmax(self.q_values(V), axis=0)

    def value_iteration(self, initial_V=None, max_iterations=1000, tolerance=0.01):
        V = np.zeros(self.nStates) if initial_V is None else np.asarray(initial_V, dtype=float)
        for iteration in range(1, max_iterations + 1):
            V_new = self.q_values(V).max(axis=0)
            epsilon = np.abs(V_new - V).max()
            V = V_new
            if epsilon < tolerance:
                break
        return V, iteration, epsilon

    def evaluate_policy(self, policy):
        states = np.arange(self.nStates)
        T_pi = self.T[policy, states, :]
        R_pi = self.R[policy, states]
        return np.linalg.solve(np.eye(self.nStates) - self.discount * T_pi, R_pi)

    def policy_iteration(self, initial_policy=None, max_iterations=1000):
        policy = np.zeros(self.nStates, dtype=int) if initial_policy is None else np.asarray(initial_policy)
        for iteration in range(1, max_iterations + 1):
            V = self.evaluate_policy(policy)
            new_policy = self.extract_policy(V)
            if np.array_equal(new_policy, policy):
                break
            policy = new_policy
        return policy, V, iteration

    def evaluate_policy_partially(self, policy, V, n_sweeps):
        states = np.arange(self.nStates)
        T_pi = self.T[policy, states, :]
        R_pi = self.R[policy, states]
        for _ in range(n_sweeps):
            V = R_pi + self.discount * T_pi @ V
        return V

    def modified_policy_iteration(self, n_eval_sweeps, initial_V=None, max_iterations=1000, tolerance=0.01):
        V = np.zeros(self.nStates) if initial_V is None else np.asarray(initial_V, dtype=float)
        for iteration in range(1, max_iterations + 1):
            policy = self.extract_policy(V)
            V_new = self.evaluate_policy_partially(policy, V, n_eval_sweeps)
            epsilon = np.abs(V_new - V).max()
            V = V_new
            if epsilon < tolerance:
                break
        return self.extract_policy(V), V, iteration, epsilon


def advertise_or_save_mdp():
    T = np.array([
        [[0.5, 0.5, 0, 0], [0, 1, 0, 0], [0.5, 0.5, 0, 0], [0, 1, 0, 0]],
        [[1, 0, 0, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0, 0.5, 0.5]],
    ])
    R = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    return MDP(T, R, discount=0.9)


def maze_mdp(intended=0.7, lateral=0.15, discount=0.95):
    """4x4 grid world. States 0-15 are cells (row-major), 16 is an absorbing end state.

    Actions: 0=up, 1=down, 2=left, 3=right. The agent moves in the intended direction
    with probability `intended` and in each perpendicular direction with probability
    `lateral`; moving into a wall leaves it in place. Cell 14 is the goal (+100, then
    the end state), cell 9 is a trap (-70), every other step costs -1.
    """
    size, goal, trap, end = 4, 14, 9, 16
    n_states = size * size + 1
    moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    perpendicular = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}

    def step(state, action):
        row, col = divmod(state, size)
        d_row, d_col = moves[action]
        new_row, new_col = row + d_row, col + d_col
        if 0 <= new_row < size and 0 <= new_col < size:
            return new_row * size + new_col
        return state

    T = np.zeros((4, n_states, n_states))
    for action in range(4):
        for state in range(size * size):
            if state == goal:
                T[action, state, end] = 1
                continue
            T[action, state, step(state, action)] += intended
            for side in perpendicular[action]:
                T[action, state, step(state, side)] += lateral
        T[action, end, end] = 1

    R = -np.ones((4, n_states))
    R[:, goal] = 100
    R[:, trap] = -70
    R[:, end] = 0
    return MDP(T, R, discount)
