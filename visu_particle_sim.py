# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:43:00 2024

Particle simulation using CUDA for gravitational force

@author: Nasser
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_particles(positions, timestep):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    ax.scatter(x, y, z, c='b', marker='o')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f'Timestep {timestep}')

    plt.show()

# Example of loading positions from a file (adapt this to your file format)
# positions = np.loadtxt('particle_positions_timestep_0.txt')

# For now, we'll generate some random data to simulate
positions = np.random.rand(1000, 3) * 1000
plot_particles(positions, 0)
