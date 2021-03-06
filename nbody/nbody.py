import json
import numpy as np
import matplotlib.pyplot as plt
from cached_property import cached_property
from datetime import datetime

class Particles:
    def __init__(self, position, velocity, mass):
        self.position = position.copy().astype(np.float)
        self.velocity = velocity.copy().astype(np.float)
        self.Npart, self.ndim = position.shape
        for i in range(self.ndim):
            setattr(self, 'xyz'[i], self.position[:,i])
            setattr(self, ('vx','vy','vz')[i], self.velocity[:,i])
        if np.isscalar(mass):
            self.mass = np.ones(self.Npart, dtype=np.float) * mass
        else:
            self.mass = mass.copy().astype(np.float)


    @property
    def kinetic_energy(self):
        if self.Npart > 1:
            return 0.5 * self.mass @ np.sum(self.velocity ** 2, axis=1)
        else:
            return 0.5 * self.mass[0] * np.sum(self.velocity ** 2)


class Mesh:
    def __init__(self, half_size, ndim, unit_length=1, Nguard=0):
        self.half_size = half_size
        self.ndim = ndim
        self.unit_length = unit_length
        self.Nguard = Nguard
        self.grid = self.make_grid(half_size + Nguard, ndim)


    @staticmethod
    def make_grid(half_size, ndim):
        return np.asarray(
            np.meshgrid(
                *np.repeat(
                    np.arange(-half_size, half_size).reshape(1,-1),
                    ndim,
                    axis=0,
                ),
                indexing="ij",
            )
        )
        

    def get_grid(self, rescale=False, copy=True, guarded=False):
        scale = self.unit_length if rescale else 1
        return_slice = (slice(None),)
        if guarded or self.Nguard == 0:
            return_slice += (slice(None),) * self.ndim
        else:
            return_slice += (slice(self.Nguard, -self.Nguard),) * self.ndim
        grid = self.grid[return_slice] * scale
        return grid.copy() if copy else grid


    def find_nearest_cell(self, position, rescale=False, guarded=False):
        scale = self.unit_length if rescale else 1
        size = self.half_size + self.Nguard if guarded else self.half_size
        rescaled_position = np.floor(position / scale) + size
        rescaled_position = np.fmin(rescaled_position, 2 * size - 1)
        return tuple(rescaled_position.astype(int))


    def get_guarded_grid(self, rescale=False, copy=True):
        return self.get_grid(rescale=rescale, copy=copy, guarded=True)

        
class ParticleMesh:
    def __init__(
        self,
        position,
        velocity,
        mass,
        half_size,
        ndim,
        unit_length=1,
        Nguard=0,
        softening=0.01,
    ):
        self.particles = Particles(position, velocity, mass)
        self.mesh = Mesh(half_size, ndim, unit_length, Nguard)
        self.softening = softening
        self.particle_locations = self.find_particle_locations()
        self.density = self.calc_density()
        self.potential = self.calc_potential()
        self.acceleration = self.calc_acceleration()


    @cached_property
    def base_potential(self):
        r_squared = np.sum(
            self.mesh.get_grid(rescale=True, guarded=True) ** 2, axis=0
        )
        r_squared[r_squared < self.softening ** 2] = self.softening ** 2
        return -1 / np.sqrt(r_squared)


    def calc_density(self):
        density = np.zeros(
            self.mesh.get_grid(guarded=True).shape[1:], dtype=np.float
        )
        for location, mass in zip(self.particle_locations, self.particles.mass):
            density[location] += mass
        return density
            

    def calc_potential(self):
        return np.fft.fftshift(
            np.fft.ifftn(
                np.fft.fftn(self.base_potential) * np.fft.fftn(self.density)
            )
        ).real


    def calc_acceleration(self):
        return _calc_acceleration(
            self.mesh.get_grid(rescale=True, guarded=True),
            self.particle_locations,
            self.potential,
        )

        
    def find_particle_locations(self):
        locations = []
        for position in self.particles.position:
            locations.append(
                self.mesh.find_nearest_cell(position, rescale=True, guarded=True)
            )
        return locations


    @property
    def potential_energy(self):
        potential = 0
        for location, mass in zip(self.particle_locations, self.particles.mass):
            potential += mass * self.potential[location]
        return potential


    @property
    def total_energy(self):
        return self.particles.kinetic_energy + self.potential_energy


    def update(self):
        print(f"Starting update at {now()}")
        self.particle_locations = self.find_particle_locations()
        print(f"Calculating density, starting at {now()}")
        self.density = self.calc_density()
        print(f"Calculating potential, starting at {now()}")
        self.potential = self.calc_potential()
        print(f"Calculating acceleration, starting at {now()}")
        self.acceleration = self.calc_acceleration()


class Nbody:
    def __init__(
        self,
        position,
        velocity,
        mass,
        half_size,
        ndim,
        unit_length=1,
        Nguard=0,
        softening=0.01,
        timestep=0.1,
        Nstep=2000,
        boundary_conditions=None,
    ):
        self.particle_mesh = ParticleMesh(
            position,
            velocity,
            mass,
            half_size,
            ndim,
            unit_length=unit_length,
            Nguard=Nguard,
            softening=softening,
        )
        self.timestep = timestep
        self.Nstep = Nstep
        self.boundary_conditions = boundary_conditions
        self.history = {
            "position": self.particle_mesh.particles.position[None,...],
            "velocity": self.particle_mesh.particles.velocity[None,...],
            "mass": self.particle_mesh.particles.mass[None,...],
            "energy": [self.particle_mesh.total_energy,],
        }
        self.info = {
            "half_size": half_size,
            "ndim": ndim,
            "unit_length": unit_length,
            "Nguard": Nguard,
            "softening": softening,
            "timestep": timestep,
            "boundary_conditions": boundary_conditions,
            "Npart": self.particle_mesh.particles.Npart,
            "steps_completed": 0,
        }
        self.info_saved = False


    def run(self, oversample=10, dump_after=100, filename=None):
        # Run the simulation and write contents to history.
        self.info["oversample"] = oversample
        Ndumps = 0
        for i in range(self.Nstep // oversample):
            for j in range(oversample):
                self.evolve()
                print(f"{i * oversample + j} steps completed at {now()}")
            if filename is not None:
                if i // dump_after != Ndumps:
                    self.save(f"{filename}_{Ndumps}")
                    Ndumps = i // dump_after
                    self.reset_history()
            self.update_history()
            self.info["steps_completed"] += oversample

        if filename is not None:
            self.save(f"{filename}_{Ndumps + 1}")


    def animate(self, oversample=10):
        # Animate a simulation that has already been run.
        pass


    def evolve(self):
        # Velocity leads position by half a step in Leapfrog integrator.
        self.particle_mesh.particles.velocity += (
            0.5 * self.particle_mesh.acceleration * self.timestep
        )
        self.particle_mesh.particles.position += (
            self.particle_mesh.particles.velocity * self.timestep
        )
        self.apply_boundary_conditions()
        
        # Recalculate density/potential/acceleration/energy.
        self.particle_mesh.update()
        self.particle_mesh.particles.velocity += (
            0.5 * self.particle_mesh.acceleration * self.timestep
        )


    def animate_realtime(self, oversample=10):
        # Run the simulation and animate simultaneously.
        pass


    def apply_boundary_conditions(self):
        # Find out if any of the particles are outside the simulation region.
        print(f"Starting to apply boundary conditions at {now()}")
        positions = self.particle_mesh.particles.position
        half_size = self.particle_mesh.mesh.half_size
        unit_length = self.particle_mesh.mesh.unit_length
        left_edge = -half_size * unit_length
        right_edge = (half_size - 1) * unit_length
        box_size = (
            self.particle_mesh.mesh.half_size * self.particle_mesh.mesh.unit_length
        )
        outside_left = positions < left_edge
        outside_right = positions > right_edge
        apply_bcs = np.any(outside_left) or np.any(outside_right)

        # If none of the particles have left the region, then there's nothing to do.
        if not apply_bcs:
            print(f"No boundary conditions to enforce. {now()}")
            return

        if self.boundary_conditions == "periodic":
            self.particle_mesh.particles.position[outside_left] = right_edge
            self.particle_mesh.particles.position[outside_right] = left_edge
        else:
            problem_particles = np.any(outside_left, axis=1) | np.any(outside_right, axis=1)
            self.particle_mesh.particles.mass[problem_particles] = 0
            self.particle_mesh.particles.position[problem_particles] = 0
            self.particle_mesh.particles.velocity[problem_particles] = 0

        print(f"Boundary conditions applied at {now()}")
        # Since we modified some positions or masses, we need to recalculate things.
        self.particle_mesh.update()
                    

    def update_history(self):
        self.history["position"] = np.concatenate(
            [self.history["position"], self.particle_mesh.particles.position[None,...]],
            axis=0,
        )
        self.history["velocity"] = np.concatenate(
            [self.history["velocity"], self.particle_mesh.particles.velocity[None,...]],
            axis=0,
        )
        self.history["mass"] = np.concatenate(
            [self.history["mass"], self.particle_mesh.particles.mass[None,...]],
            axis=0,
        )
        self.history["energy"].append(self.particle_mesh.total_energy)


    def reset_history(self):
        self.history["position"] = self.particle_mesh.particles.position[None,...]
        self.history["velocity"] = self.particle_mesh.particles.velocity[None,...]
        self.history["mass"] = self.particle_mesh.particles.mass[None,...]
        self.history["energy"] = [self.particle_mesh.total_energy]


    def save(self, filename):
        np.savez(filename, **self.history)
        if not self.info_saved:
            with open(f"{filename[:-2]}_info.json", "w") as fp:
                json.dump(self.info, fp)
            self.info_saved = True


def calc_gradient(potential, location, resolution):
    gradient = np.zeros(potential.ndim)
    for axis in range(potential.ndim):
        location_A = list(location)
        location_B = list(location)
        this_index = location[axis]

        if this_index in (0, potential.shape[axis] - 1):
            dx = resolution
            if this_index == 0:
                location_B[axis] += 1  # Use a right difference at the left edge.
            else:
                location_A[axis] -= 1  # Use a left difference at the right edge.

        else:
            # Use a central difference otherwise.
            dx = 2 * resolution
            location_A[axis] -= 1
            location_B[axis] += 1

        location_A = tuple(location_A)
        location_B = tuple(location_B)
        gradient[axis] = (potential[location_B] - potential[location_A]) / dx
    return gradient


def _calc_acceleration(grid, locations, potential):
    acceleration = np.zeros((len(locations), grid.shape[0]), dtype=np.float)
    resolution = np.mean(np.diff(grid)) * grid.shape[0]
    for i, location in enumerate(locations):
        acceleration[i] = -calc_gradient(potential, location, resolution)
    return acceleration


def now():
    now = datetime.now()
    return f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
