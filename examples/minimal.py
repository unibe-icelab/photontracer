import numpy as np
from photontracer import Simulation, Material, REFRACTIVE, ParallelRayGenerator, MeshGeometry, OutputType
import photontracer
print("PhotonTracer version:", photontracer.__version__)

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

sim = Simulation(
    gpu_id=0,
    enable_validation_mode=True,
    optix_logging_level=4)

sim.geometry = MeshGeometry(vertices, faces)

sim.number_of_rays = 1

sim.materials = [Material(REFRACTIVE, 1+0j), Material(REFRACTIVE, 1.33+1e-6j)]
sim.wavelength_um = wavelength
sim.ray_generator = ParallelRayGenerator(
    origin=origin,
    direction=direction,
    offset_radius=0
)
sim.outputs = [OutputType.SCATTERING_COUNT]

sim.run()

scattering_count = sim.get_output_buffer(OutputType.SCATTERING_COUNT)
print(scattering_count.mean())
