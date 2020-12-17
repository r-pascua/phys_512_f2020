import glob
import sys
from datetime import datetime as dt

import numpy as np

import nbody

if __name__=="__main__":
    try:
        part = sys.argv[1]
    except IndexError:
        part = "3a"
    ndim = 3
    grid_half_size = 50
    grid_length = 1
    Nguard = 5
    unit_length = grid_length / grid_half_size
    soften_scale = 3
    softening = soften_scale * unit_length
    if part == 1:
        boundary_conditions = None
        masses = 1
        positions = np.zeros((1, ndim), dtype=np.float)
        velocities = np.zeros_like(positions)
        timestep = 0.1
        Nstep = 100
        save_dir = "./part_1"
    elif part == 2:
        boundary_conditions = None
        masses = 1
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        phi = np.random.uniform(0, 2 * np.pi)
        radius = 10.23 * unit_length
        omega = np.sqrt(masses / (4 * radius ** 3))
        r0 = radius * np.array([np.cos(phi), np.sin(phi), 0])
        v0 = radius * omega * np.array([-np.sin(phi), np.cos(phi), 0])
        rot_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), np.sin(theta)],
                [0, -np.sin(theta), np.cos(theta)],
            ]
        )
        r0 = rot_mat @ r0
        v0 = rot_mat @ v0
        positions = np.array([r0, -r0])
        velocities = np.array([v0, -v0])
        C = 0.05
        timestep = radius * omega / (C * unit_length)
        Nstep = 2 * int(np.ceil(2 * np.pi / (timestep * omega)))
        save_dir = "./part_2"
        print(Nstep)
    elif part.lower() in ("3a", "3b"):
        boundary_conditions = "nonperiodic" if part.lower() == "3a" else "periodic"
        Npart = 300000
        masses = 1
        positions = np.random.uniform(-grid_length, grid_length, (Npart, ndim))
        velocities = np.zeros((Npart, ndim), dtype=np.float)
        timestep = 0.00001
        Nstep = 10000
        save_dir = "./part_3/{boundary_conditions}"
    else:
        boundary_conditions = "periodic"
        # Generate a density field from a scale-invariant power spectrum realization.
        k_modes = 2 * np.pi * np.fft.fftfreq(2 * grid_half_size, unit_length)
        k_mode_mesh = np.array(np.meshgrid([k_modes,] * ndim, indexing="ij"))
        k_mode_amps = np.linalg.norm(k_mode_mesh, axis=0)
        power = np.zeros((2 * grid_half_size,) * ndim, dtype=np.float)
        power[k_mode_amps > 0] = k_mode_amps[k_mode_amps > 0] ** -ndim
        white_noise = np.random.normal(size=(2 * grid_half_size,) * ndim)
        density = np.abs(
            np.fft.ifftn(
                np.fft.fftn(white_noise) * np.sqrt(power)
            )
        )
        # Place particles at cell centers, then assign masses according to density.
        mesh = nbody.Mesh(grid_half_size, ndim, unit_length, Nguard)
        grid = mesh.get_grid(rescale=False, guarded=False)
        left_slice = (slice(None,-1),) * ndim
        right_slice = (slice(1,None),) * ndim
        cell_centers = np.array(
            [
                0.5 * (grid[i][left_slice] + grid[i][right_slice])
                for i in range(ndim)
            ]
        )
        indices = np.indices(cell_centers[0].shape)
        indices = list(zip(*[indices[i].flat for i in range(ndim)]))
        masses = np.array([density[inds] for inds in indices])
        positions = np.array([zip(*[cell_centers[i].flat for i in range(ndim)])])
        velocities = np.zeros_like(positions)
        timestep = 0.00001
        Nstep = 10000
        save_dir = "./part_4"

    existing_simulation_files = sorted(
        glob.glob(f"{save_dir}/*.npz"), key=lambda f: int(f[-7])
    )
    if existing_simulation_files:
        simulation_number = existing_simulation_files[-1][-7]
    else:
        simulation_number = 0
    oversample = 10
    dump_after = 10
    filename = f"{save_dir}/simulation_{simulation_number}"
    print("Setting up simulation...")
    nbody = nbody.Nbody(
        positions,
        velocities,
        masses,
        grid_half_size,
        ndim,
        unit_length=unit_length,
        Nguard=Nguard,
        softening=softening,
        timestep=timestep,
        Nstep=Nstep,
        boundary_conditions=boundary_conditions,
    )
    now = dt.now()
    print(f"Starting simulation run at {now.hour}:{now.minute}:{now.second}")
    nbody.run(oversample=oversample, dump_after=dump_after, filename=filename)
