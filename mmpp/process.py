import numpy as np
import pandas as pd
from typing import List

class MarkovModulatedPoissonProcess:
    def __init__(self, generator_matrix: np.ndarray, intensities: List):
        """
        :param generator_matrix: The generator matrix (Q matrix) of the CTMC.
        :param intensities: List of state names corresponding to the states of the CTMC.
        """
        self.generator_matrix = generator_matrix
        self.intensities = intensities
        self.idx_map = {state: i for i, state in enumerate(intensities)}
        self.num_states = len(intensities)

        if self.generator_matrix.shape != (self.num_states, self.num_states):
            raise ValueError(
                "The generator matrix must be square with dimensions matching the number of states."
            )

        # Ensure rows of the generator matrix sum to zero
        if not np.allclose(self.generator_matrix.sum(axis=1), 0):
            raise ValueError("Rows of the generator matrix must sum to zero.")

    def simulate(
        self, start_intensity: float, end_time: float, seed=None
    ) -> pd.DataFrame:
        """
        Simulate the MMPP.

        :param start_intensity: The starting state of the simulation.
        :param end_time: The end time of the simulation.
        :param seed: Optional random seed for reproducibility.
        :return: A Pandas DataFrame containing the simulation history.
        """
        if start_intensity not in self.intensities:
            raise ValueError("Invalid start state. Must be one of the defined states.")

        rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

        current_intensity_idx = self.idx_map[
            start_intensity
        ]  # work in index domain for simulation
        current_time = 0
        count = 0
        history = [(current_time, count, start_intensity)]

        while current_time < end_time:
            rates = self.generator_matrix[current_intensity_idx]

            transition_times = rng.exponential(np.abs(1 / rates))
            transition_times[current_intensity_idx] = np.inf

            if np.all(transition_times == np.inf):
                if len(history) > 1:
                    history[-1][0] = np.inf
                break

            next_intensity_idx = transition_times.argmin()
            next_intensity_time = current_time + transition_times[next_intensity_idx]
            time_gap = transition_times[next_intensity_idx]

            time_to_intermediate_arrival = np.random.exponential(
                1 / self.intensities[current_intensity_idx]
            )

            while time_to_intermediate_arrival < time_gap:
                count += 1
                current_time += time_to_intermediate_arrival
                history.append(
                    (current_time, count, self.intensities[current_intensity_idx])
                )
                time_gap -= time_to_intermediate_arrival
                time_to_intermediate_arrival = np.random.exponential(
                    1 / self.intensities[current_intensity_idx]
                )

            current_time = next_intensity_time
            current_intensity_idx = next_intensity_idx

            history.append(
                (current_time, count, self.intensities[current_intensity_idx])
            )

        df = pd.DataFrame(history, columns=["Time", "Count", "CTMC State"])
        return df
