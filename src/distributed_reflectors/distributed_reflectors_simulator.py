import numpy as np

from distributed_reflectors.boundries import BoxBoundry

EPS = 10 ** -5


class DRSimulator:
    def __init__(
        self,
        boundry: BoxBoundry,
        reflectors: list,
        num_particles: int,
        num_allowed_collisions: int,
        particle_amplitudes: np.ndarray,
    ):

        self.boundry = boundry
        self.reflectors = reflectors
        self.num_particles = num_particles
        self.num_allowed_collisions = num_allowed_collisions
        self.particle_amplitudes = particle_amplitudes

    def _initialize_particles(self):

        particle_coordinates = np.zeros((self.num_particles, 3))

        particle_coordinates[:, 0] = self._get_coordinate_samples(
            self.particle_amplitudes[0, :]
        )
        particle_coordinates[:, 1] = self._get_coordinate_samples(
            self.particle_amplitudes[1, :]
        )
        particle_coordinates[:, 2] = self._get_coordinate_samples(
            self.particle_amplitudes[2, :]
        )

        return particle_coordinates

    def _get_coordinate_samples(self, amplitude: np.ndarray) -> np.ndarray:
        return amplitude[0] + np.random.rand(self.num_particles) * (
            amplitude[1] - amplitude[0]
        )

    def set_reflector_angles(self, angles: np.ndarray):
        self.reflectors = [
            reflector.set_angle(angle)
            for reflector, angle in zip(self.reflectors, angles)
        ]

    def simulate(self, verbose: bool = False):
        particle_coordinates = self._initialize_particles()
        num_particles = particle_coordinates.shape[0]
        trajectories = np.zeros((self.num_allowed_collisions, num_particles, 2))

        for colission in range(self.num_allowed_collisions):

            # get the indices of the active particles

            # TODO reimplement this using the boundry class
            active_particles_idx = get_active_particles_idx(particle_coordinates, H, W)
            num_active_particles = len(active_particles_idx)
            if num_active_particles == 0:
                return trajectories[:colission, :, :]

            # run active particles through system for one collision
            active_particle_coordinates = particle_coordinates[active_particles_idx, :]
            active_particle_coordinates = self.run_one_collision(
                active_particle_coordinates
            )

            # save trajectories in trajectories array
            particle_coordinates[active_particles_idx, :] = active_particle_coordinates
            trajectories[colission, :, :] = particle_coordinates[:, :2]

        if verbose:
            print(trajectories)

        return trajectories

    def run_one_collision(self, active_particle_coordinates):

        num_active_particles = active_particle_coordinates.shape[0]
        all_colisions = np.zeros(num_active_particles)

        # run through all reflectors
        for reflector in self.reflectors:

            # check which particles collided with the current reflector
            particle_collisions = reflector.will_collide(active_particle_coordinates)

            # get new coordinates for the particles that collided with the reflector
            active_particle_coordinates = reflector.get_new_coordinates(
                active_particle_coordinates, particle_collisions
            )

            # count all reflections that occured
            all_colisions += particle_collisions

        # get final coordinates of particles that passed through all the reflectors
        active_particle_coordinates = get_final_coordinates(
            H, W, active_particle_coordinates, all_colisions
        )

        return active_particle_coordinates

    def get_active_particles_idx(particle_coordinates, H, W):
        x_boundry_not_reached = np.abs(particle_coordinates[:, 0]) != W / 2
        y_boundry_not_reached = np.abs(particle_coordinates[:, 1]) != H / 2
        active_particles = np.logical_and(x_boundry_not_reached, y_boundry_not_reached)
        active_particles_idx = np.nonzero(active_particles)[0]
        return active_particles_idx
