import glob
import itertools
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def make_plot(
    masked_positions, box_size, times=None, energy=None, Emin=None, Emax=None
):
    corners = get_corners(box_size)

    # Just plot the box with the particles in it.
    if times is None or energy is None:
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(211, projection="3d")
        ax = plot_box_edges(ax, corners)
        ax.scatter(*masked_positions.T, marker="o", s=0.001)
    else:  # Plot the total energy along with the box.
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6,8), dpi=100)
        spec = axes[1].get_gridspec()
        for ax in axes[:-1]:
            ax.remove()
        box = fig.add_subplot(spec[:-1,0], projection="3d")
        box = plot_box_edges(box, corners)
        box.scatter(*masked_positions.T, marker="o", s=0.001)
        if Emin is not None and Emax is not None:
            axes[-1].set_ylim(Emin, Emax)
        axes[-1].plot(times, energy)
        axes[-1].text(
            0.05,
            0.9,
            "$E(t)$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=axes[-1].transAxes,
        )
    return fig


def plot_box_edges(axes, corners):
    for edge in itertools.combinations(corners, 2):
        edge = np.array(edge)
        if np.sum(edge[1] - edge[0] == 0) == 2:
            axes.plot(*edge.T, color="k", lw=2)
    axes.set_axis_off()
    return axes


def get_corners(box_size):
    corners = np.ones((8,3), dtype=np.float) * box_size
    index = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for m, n in enumerate((i,j,k)):
                    corners[index][m] *= (-1) ** n
                index += 1
    return corners


if __name__=="__main__":
    data_path, basename = os.path.split(sys.argv[1])
    filenames = sorted(
        glob.glob(f"{os.path.join(data_path, basename)}_?.npz")
    )
    try:
        destination = sys.argv[2]
    except IndexError:
        destination = "./"

    # Load the data and simulation info.
    frame_number = 0
    with open(f"{os.path.join(data_path, basename)}_info.json", "r") as f:
        info = json.load(f)
    modified_timestep = info["timestep"] * info["oversample"]
    box_size = info["half_size"] * info["unit_length"]
    for chunk_number, filename in enumerate(filenames):
        data = np.load(filename)

        # Prepare the data for plotting.
        positions = data["position"]
        if chunk_number == 0:
            energy = data["energy"]
            time = np.arange(positions.shape[0]) * modified_timestep
        else:
            energy = np.append(energy, data["energy"])
            time = np.append(
                time,
                np.arange(positions.shape[0]) * modified_timestep + time.max()
            )
        
        # Make sure to mask the points outside the box.
        outside_left = positions < -box_size
        outside_right = positions >= box_size
        mask = outside_left | outside_right
        if np.any(mask):
            print("particles found outside box")
        for i in range(3):
            mask[...,i] = np.any(mask, axis=-1)
        masked_positions = np.ma.MaskedArray(positions, mask)

        # Now actually plot the data and save the images.
        for i in range(positions.shape[0]):
            print(f"Creating frame {frame_number}...")
            fig = make_plot(
                masked_positions[i],
                box_size,
                time[:frame_number],
                energy[:frame_number],
                energy.min(),
                energy.max(),
            )
            fig.savefig(
                f"{os.path.join(destination, basename)}_frame_{frame_number}.png",
                dpi=100,
            )
            fig.clf()
            plt.close(fig)
            frame_number += 1
