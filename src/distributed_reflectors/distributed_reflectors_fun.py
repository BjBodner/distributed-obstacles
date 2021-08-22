import numpy as np

# from src.distributed_reflectors.boundries import BoxBoundry
# # from src.distributed_reflectors.distributed_reflectors_simulator import \
# #     DRSimulator
# from src.distributed_reflectors.reflector import get_reflectors

EPS = 10 ** -5
PI = np.pi


# class DistributedReflectors:
#     def __init__(
#         self,
#         system_height: float = 5.0,
#         system_width: float = 5.0,
#         particle_amplitudes: list = [[-2.0, -2.0], [-2, 2], [0, PI]],
#         reflector_position_amplitudes: list = [2.0, 2.0, PI],
#         reflector_length: float = 1.0,
#         num_reflectors: int = 5,
#         num_particles: int = 1000,
#         num_allowed_collisions: int = 20,
#         random_state: int = None,
#     ) -> None:

#         boundry = BoxBoundry(system_height, system_width)
#         self.x_range, self.y_range = self.boundry.get_ranges()
#         reflectors = get_reflectors(
#             self.x_range,
#             self.y_range,
#             reflector_position_amplitudes,
#             reflector_length,
#             num_reflectors,
#             random_state,
#         )
#         self.num_particles = num_particles
#         self.simulator = DRSimulator(
#             boundry,
#             reflectors,
#             num_particles,
#             num_allowed_collisions,
#             particle_amplitudes,
#         )

# def calculate_loss(trajectories, H, W):
#     final_coordinates = trajectories[-1, :, :]
#     square_norms = np.linalg.norm(final_coordinates, axis=1) ** 2
#     return np.mean(square_norms)

# def fun(self, angles: np.ndarray):

#     # run simulation with the angle parameters
#     self.simulator.set_reflector_angles(angles)
#     trajectories = self.simulator.simulate()

#     # calculate loss
#     final_coordinates = trajectories[-1, :, :]
#     square_norms = np.linalg.norm(final_coordinates, axis=1) ** 2
#     loss = np.mean(square_norms)

#     # calculate penalty
#     penalty = np.sum((np.mod(angles, PI) - angles) ** 2)

#     return loss + penalty
