import numpy as np
import photontracer as phtr

print("PhotonTracer version:", phtr.__version__)

wavelength = 0.6

# cube vertices and faces
vertices = np.array([
    [-50, -50, 0],
    [50, -50, 0],
    [50,  50, 0],
    [-50,  50, 0],
    [-50, -50, 100],
    [50, -50, 100],
    [50,  50, 100],
    [-50,  50, 100],
], dtype=np.float32)

faces = np.array([
    [0, 2, 1], [0, 3, 2],  # Bottom face
    [4, 6, 5], [4, 7, 6],  # Top face
    [0, 4, 5], [0, 5, 1],  # Back face
    [2, 7, 3], [2, 6, 7],  # Front face
    [1, 5, 6], [1, 6, 2],  # Right face
    [3, 7, 4], [3, 4, 0],  # Left face
], dtype=np.uint32)

direction = np.array([0, 0, -1])
origin = -direction * 200

sim = phtr.Simulation()

sim.geometry = phtr.MeshGeometry(vertices, faces)

sim.materials = [
    phtr.Material(phtr.MaterialType.REFRACTIVE, 1+0j),  # vacuum
    phtr.Material(phtr.MaterialType.REFRACTIVE, 1.33+1e-6j),  # water
]
sim.wavelength_um = wavelength
sim.ray_generator = phtr.ParallelRayGenerator(
    number_of_rays=1,
    origin=origin,
    direction=direction,
    offset_radius=0
)
sim.outputs = [phtr.OutputType.SCATTERING_COUNT]

sim.run()

scattering_count = sim.get_output_buffer(phtr.OutputType.SCATTERING_COUNT)
print(scattering_count.mean())
