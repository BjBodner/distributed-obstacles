from typing import Union

import numpy as np


def parse_coordinates(
    particle_coordinates: np.ndarray, selected_indices: Union[np.ndarray, list] = None
) -> np.ndarray:
    if selected_indices is None:
        selected_indices = list(range(particle_coordinates.shape[0]))
    particles_x = particle_coordinates[selected_indices, 0]
    particles_y = particle_coordinates[selected_indices, 1]
    particles_angles = particle_coordinates[selected_indices, 2]

    return particles_x, particles_y, particles_angles
