import numpy as np
import nbody
from datetime import datetime as dt

if __name__=="__main__":
    ndim = 3
    grid_half_size = 100
    grid_length = 1
    Nguard = 10
    unit_length = grid_length / grid_half_size
    softening = 0.03
    masses = 1
    Npart = 300000
    positions = np.random.uniform(-grid_length, grid_length, (Npart, ndim))
    velocities = np.zeros((Npart, ndim), dtype=np.float)
    boundary_conditions = "periodic"
    Nstep = 1000
    oversample = 10
    timestep = 0.0001
    dump_after = 50
    filename = f"./part_3/{boundary_conditions}/simulation_1"
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
