import numpy as np
import matplotlib.pyplot as plt
import nbody_better

if __name__=="__main__":
    ndim = 2
    grid_half_size = 100
    grid_length = 4
    Nguard = 10
    unit_length = grid_length / grid_half_size
    softening = 0.03
    setup = "random"

    if setup == "circular":
        masses = np.ones(2)
        phi = np.random.uniform(0, np.pi)
        omega = 1
        radius = (masses[0] / (4 * omega ** 2)) ** (1/3)
        r0 = radius * np.exp(1j * phi)
        v0 = 1j * r0 * omega
        pos = np.array([[r0.real, r0.imag], [-r0.real, -r0.imag]])
        vel = np.array([[v0.real, v0.imag], [-v0.real, -v0.imag]])
        expected_path = radius * np.exp(1j * (omega * dt * np.arange(Nstep) + phi))
        dt = 0.01
        quiver = True
    elif setup == "bc":
        masses = np.ones(1)
        pos = np.array([[0,0],], dtype=np.float)
        vel = np.array([[1,1],], dtype=np.float)
        dt = 0.1
        quiver = False
    elif setup == "random":
        Npart = 100
        masses = np.random.uniform(1, 10, Npart)
        pos = np.random.uniform(-grid_length, grid_length, (Npart, ndim))
        vel = np.zeros((Npart, ndim), dtype=np.float)
        dt = 0.0005
        quiver = False
    
    Nstep = 5000
    oversamp = 10
    boundary_conditions = "periodic"
    nbody = nbody_better.Nbody(
        pos,
        vel,
        masses,
        grid_half_size,
        ndim,
        unit_length=unit_length,
        Nguard=Nguard,
        softening=softening,
        timestep=dt,
        Nstep=Nstep,
        boundary_conditions=boundary_conditions,
    )
    if Nguard:
        this_slice = (slice(Nguard,-Nguard),) * ndim
    else:
        this_slice = (slice(None),) * ndim

    plt.ion()
    plt.plot(pos[:,0], pos[:,1], 'r.')
    plt.xlim(-grid_length, grid_length)
    plt.ylim(-grid_length, grid_length)
    plt.show()
    for i in range(Nstep//oversamp):
        for j in range(oversamp):
            nbody.evolve()
            nbody.apply_boundary_conditions()
        
        nbody.update_history()
        plt.clf()
        plt.plot(
            nbody.particle_mesh.particles.x,
            nbody.particle_mesh.particles.y,
            'r.',
        )
        if setup == "circular":
            plt.plot(expected_path.real, expected_path.imag, color='k', ls=':')
        plt.xlim(-grid_length, grid_length)
        plt.ylim(-grid_length, grid_length)
        plt.imshow(
            nbody.particle_mesh.potential[this_slice].T,
            extent=[-grid_length, grid_length, grid_length, -grid_length],
            cmap="viridis",
        )
        plt.colorbar()
        if quiver:
            plt.quiver(
                nbody.particle_mesh.particles.x,
                nbody.particle_mesh.particles.y,
                nbody.particle_mesh.acceleration[:,0],
                nbody.particle_mesh.acceleration[:,1],
                color="forestgreen",
            )
            plt.quiver(
                nbody.particle_mesh.particles.x,
                nbody.particle_mesh.particles.y,
                nbody.particle_mesh.particles.vx,
                nbody.particle_mesh.particles.vy,
                color="darkorange",
            )
        plt.text(
            0.5 * grid_length,
            0.9 * grid_length,
            f"t = {i * dt * oversamp:.3f}",
            horizontalalignment="left",
        )
        plt.text(
            -0.9 * grid_length,
            0.9 * grid_length,
            f"E = {nbody.particle_mesh.total_energy:.3f}",
            horizontalalignment="left",
        )
        plt.pause(0.0001)

    plt.close()
    nbody.save("./test_data_nbody.npz")
