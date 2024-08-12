#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_PARTICLES 1000
#define TIMESTEPS 1000
#define G 6.67430e-11f // Gravitational constant
#define DT 0.01f       // Time step

struct Particle {
    float3 position;
    float3 velocity;
    float mass;
};

__device__ float3 compute_gravitational_force(Particle p1, Particle p2) {
    float3 r;
    r.x = p2.position.x - p1.position.x;
    r.y = p2.position.y - p1.position.y;
    r.z = p2.position.z - p1.position.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 1e-9f; // Add small value to avoid division by zero
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = rsqrtf(distSixth);

    float3 force;
    force.x = G * p1.mass * p2.mass * r.x * invDistCube;
    force.y = G * p1.mass * p2.mass * r.y * invDistCube;
    force.z = G * p1.mass * p2.mass * r.z * invDistCube;

    return force;
}

__global__ void update_particles(Particle *particles, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_particles) {
        Particle p = particles[idx];
        float3 net_force = make_float3(0.0f, 0.0f, 0.0f);

        for (int i = 0; i < num_particles; ++i) {
            if (i != idx) {
                float3 force = compute_gravitational_force(p, particles[i]);
                net_force.x += force.x;
                net_force.y += force.y;
                net_force.z += force.z;
            }
        }

        p.velocity.x += (net_force.x / p.mass) * DT;
        p.velocity.y += (net_force.y / p.mass) * DT;
        p.velocity.z += (net_force.z / p.mass) * DT;

        p.position.x += p.velocity.x * DT;
        p.position.y += p.velocity.y * DT;
        p.position.z += p.velocity.z * DT;

        particles[idx] = p;
    }
}

void save_positions_to_file(Particle *particles, int num_particles, int timestep) {
    char filename[64];
    sprintf(filename, "particle_positions_timestep_%d.txt", timestep);
    FILE *file = fopen(filename, "w");

    for (int i = 0; i < num_particles; ++i) {
        fprintf(file, "Particle %d: Position (%f, %f, %f)\n", i, 
                particles[i].position.x, particles[i].position.y, particles[i].position.z);
    }

    fclose(file);
}

int main() {
    Particle *h_particles, *d_particles;
    size_t size = NUM_PARTICLES * sizeof(Particle);

    h_particles = (Particle *)malloc(size);
    cudaMalloc(&d_particles, size);

    // Initialize particles randomly
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_particles[i].position = make_float3(rand() % 1000, rand() % 1000, rand() % 1000);
        h_particles[i].velocity = make_float3(0.0f, 0.0f, 0.0f);
        h_particles[i].mass = rand() % 100 + 1;
    }

    cudaMemcpy(d_particles, h_particles, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    for (int t = 0; t < TIMESTEPS; ++t) {
        printf("Running timestep %d\n", t); // Print the current timestep
        update_particles<<<blocksPerGrid, threadsPerBlock>>>(d_particles, NUM_PARTICLES);
        cudaMemcpy(h_particles, d_particles, size, cudaMemcpyDeviceToHost);

        // Save particle positions to file for visualization
        save_positions_to_file(h_particles, NUM_PARTICLES, t);
    }

    cudaFree(d_particles);
    free(h_particles);

    return 0;
}
