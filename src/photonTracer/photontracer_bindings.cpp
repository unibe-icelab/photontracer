// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

#include "simulation.h"
#include "geometries/mesh_geometry.h"
#include "geometries/instance_geometry.h"
#include "photontracer.h" // For enums such as GeometryType, MaterialType, Material, etc.

namespace py = pybind11;

float3 toFloat3(py::object obj, const std::string &name)
{
    float3 result;

    if (py::isinstance<py::array>(obj))
    {
        auto arr = py::cast<py::array_t<float>>(obj);
        if (arr.size() != 3)
        {
            throw py::value_error(name + " array must have exactly 3 elements");
        }
        auto ptr = arr.unchecked<1>();
        result.x = ptr(0);
        result.y = ptr(1);
        result.z = ptr(2);
    }
    else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj))
    {
        auto seq = py::cast<py::sequence>(obj);
        if (seq.size() != 3)
        {
            throw py::value_error(name + " sequence must have exactly 3 elements");
        }
        result.x = py::cast<float>(seq[0]);
        result.y = py::cast<float>(seq[1]);
        result.z = py::cast<float>(seq[2]);
    }
    else
    {
        throw py::type_error(name + " must be array, tuple, or list of 3 floats");
    }

    return result;
}

float4 toFloat4(py::object obj, const std::string &name)
{
    float4 result;

    if (py::isinstance<py::array>(obj))
    {
        auto arr = py::cast<py::array_t<float>>(obj);
        if (arr.size() != 4)
        {
            throw py::value_error(name + " array must have exactly 4 elements");
        }
        auto ptr = arr.unchecked<1>();
        result.x = ptr(0);
        result.y = ptr(1);
        result.z = ptr(2);
        result.w = ptr(3);
    }
    else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj))
    {
        auto seq = py::cast<py::sequence>(obj);
        if (seq.size() != 4)
        {
            throw py::value_error(name + " sequence must have exactly 4 elements");
        }
        result.x = py::cast<float>(seq[0]);
        result.y = py::cast<float>(seq[1]);
        result.z = py::cast<float>(seq[2]);
        result.w = py::cast<float>(seq[3]);
    }
    else
    {
        throw py::type_error(name + " must be array, tuple, or list of 4 floats");
    }

    return result;
}

py::object toPyObject(const float4 &vec)
{
    return py::make_tuple(vec.x, vec.y, vec.z, vec.w);
}

py::object toPyObject(const float3 &vec)
{
    return py::make_tuple(vec.x, vec.y, vec.z);
}

PYBIND11_MODULE(photontracer_bindings, m)
{
    m.doc() = "Python bindings for the Photontracer simulation and geometry functionality";

    // Expose enums.
    py::enum_<GeometryType>(m, "GeometryType")
        .value("MESH", MESH)
        .value("MESH_INSTANCED", MESH_INSTANCED)
        .export_values();

    py::enum_<LengthUnit>(m, "LengthUnit")
        .value("MICRO_METER", LengthUnit::MICRO_METER)
        .value("MILLI_METER", LengthUnit::MILLI_METER)
        .value("METER", LengthUnit::METER)
        .export_values();

    py::enum_<MaterialType>(m, "MaterialType")
        .value("DIFFUSE", DIFFUSE)
        .value("REFRACTIVE", REFRACTIVE)
        .value("REFLECTIVE", REFLECTIVE)
        .value("VOLUME_SCATTERING", VOLUME_SCATTERING)
        .export_values();

    py::enum_<OutputType>(m, "OutputType")
        .value("LAST_DIRECTION", OutputType::LAST_DIRECTION)
        .value("LAST_POSITION", OutputType::LAST_POSITION)
        .value("RAY_STATE", OutputType::RAY_STATE)
        .value("LAST_MEDIUM_ID", OutputType::LAST_MEDIUM_ID)
        .value("SCATTERING_COUNT", OutputType::SCATTERING_COUNT)
        .value("NUMBER_OF_WARNINGS", OutputType::NUMBER_OF_WARNINGS)
        .value("STOKES_VECTOR", OutputType::STOKES_VECTOR)
        .value("STOKES_VECTOR_IN", OutputType::STOKES_VECTOR_IN)
        .value("OPTICAL_PATH_LENGTH", OutputType::OPTICAL_PATH_LENGTH)
        .value("SOURCE_DIRECTION", OutputType::SOURCE_DIRECTION)
        .value("SOURCE_POSITION", OutputType::SOURCE_POSITION)
        .value("SCATTERING_ANGLE", OutputType::SCATTERING_ANGLE)
        .value("Q_MINUS_AXIS_IN", OutputType::Q_MINUS_AXIS_IN)
        .value("LOGS", OutputType::LOGS)
        .value("LOG_OFFSETS", OutputType::LOG_OFFSETS)
        .value("DIRECTION_HISTOGRAM_HEALPIX", OutputType::DIRECTION_HISTOGRAM_HEALPIX)
        .export_values();

    // Bind RefractiveIndex with support for Python complex numbers.
    py::class_<RefractiveIndex>(m, "RefractiveIndex")
        .def(py::init<>())
        .def(py::init<float, float>(), py::arg("r"), py::arg("i"))
        // Add constructor that takes a Python complex number
        .def(py::init([](const std::complex<float> &c)
                      {
            RefractiveIndex ri;
            ri.r = c.real();
            ri.i = c.imag();
            return ri; }),
             py::arg("complex_value"))
        // Add constructor that takes a Python complex number (double precision)
        .def(py::init([](const std::complex<double> &c)
                      {
            RefractiveIndex ri;
            ri.r = static_cast<float>(c.real());
            ri.i = static_cast<float>(c.imag());
            return ri; }),
             py::arg("complex_value"))
        .def_readwrite("r", &RefractiveIndex::r)
        .def_readwrite("i", &RefractiveIndex::i)
        // Add property to get/set as complex number
        .def_property("complex",
                      // Getter: return as Python complex
                      [](const RefractiveIndex &ri)
                      { return std::complex<float>(ri.r, ri.i); },
                      // Setter: accept Python complex
                      [](RefractiveIndex &ri, const std::complex<float> &c)
                      { 
                ri.r = c.real(); 
                ri.i = c.imag(); })
        .def("__repr__", [](const RefractiveIndex &ri)
             { return "<RefractiveIndex (real=" + std::to_string(ri.r) +
                      ", imag=" + std::to_string(ri.i) + ")>"; });

    // Bind the Diffuse variant inside MaterialProperties.
    py::class_<MaterialProperties::Diffuse>(m, "Diffuse")
        .def(py::init<>())
        .def_readwrite("albedo", &MaterialProperties::Diffuse::albedo)
        .def("__repr__", [](const MaterialProperties::Diffuse &l)
             { return "<Diffuse albedo=" + std::to_string(l.albedo) + ">"; });

    // Bind the Refractive variant inside MaterialProperties.
    py::class_<MaterialProperties::Refractive>(m, "Refractive")
        .def(py::init<>())
        .def_readwrite("refractive_index", &MaterialProperties::Refractive::refractiveIndex)
        .def("__repr__", [](const MaterialProperties::Refractive &r)
             { return "<Refractive refractive_index=" + std::to_string(r.refractiveIndex.r) +
                      " + " + std::to_string(r.refractiveIndex.i) + "i>"; });

    // Bind the Reflective variant inside MaterialProperties.
    py::class_<MaterialProperties::Reflective>(m, "Reflective")
        .def(py::init<>())
        .def_readwrite("reflectivity", &MaterialProperties::Reflective::reflectivity)
        .def_readwrite("fuzziness", &MaterialProperties::Reflective::fuzziness)
        .def("__repr__", [](const MaterialProperties::Reflective &r)
             { return "<Reflective reflectivity=" + std::to_string(r.reflectivity) +
                      ", fuzziness=" + std::to_string(r.fuzziness) + ">"; });

    // Bind the VolumeScattering variant inside MaterialProperties.
    py::class_<MaterialProperties::VolumeScattering>(m, "VolumeScattering")
        .def(py::init<>())
        .def_readwrite("refractive_index", &MaterialProperties::VolumeScattering::refractiveIndex)
        .def_readwrite("scattering_coefficient", &MaterialProperties::VolumeScattering::scatteringCoefficient)
        .def_readwrite("asymetry_parameter", &MaterialProperties::VolumeScattering::asymetryParameter)
        .def("__repr__", [](const MaterialProperties::VolumeScattering &v)
             { return "<VolumeScattering refractive_index=" + std::to_string(v.refractiveIndex.r) +
                      " + " + std::to_string(v.refractiveIndex.i) + "i, scattering_coefficient=" +
                      std::to_string(v.scatteringCoefficient) + ", asymetry_parameter=" + std::to_string(v.asymetryParameter) + ">"; });

    // Bind MaterialProperties as a whole, providing property getters/setters for each variant.
    py::class_<MaterialProperties>(m, "MaterialProperties")
        .def(py::init([]()
                      {
            MaterialProperties mp = {};
            return mp; }))
        .def_property("diffuse",
                      // Getter: return a reference to the Diffuse struct.
                      [](MaterialProperties &mp) -> MaterialProperties::Diffuse &
                      { return mp.diffuse; },
                      // Setter: set the diffuse field.
                      [](MaterialProperties &mp, const MaterialProperties::Diffuse &l)
                      { mp.diffuse = l; }, "Access Diffuse properties")
        .def_property("refractive",
                      // Getter: return a reference to the Refractive struct.
                      [](MaterialProperties &mp) -> MaterialProperties::Refractive &
                      { return mp.refractive; },
                      // Setter: set the refractive field.
                      [](MaterialProperties &mp, const MaterialProperties::Refractive &r)
                      { mp.refractive = r; }, "Access Refractive properties")
        .def_property("reflective",
                      // Getter: return a reference to the Reflective struct.
                      [](MaterialProperties &mp) -> MaterialProperties::Reflective &
                      { return mp.reflective; },
                      // Setter: set the reflective field.
                      [](MaterialProperties &mp, const MaterialProperties::Reflective &r)
                      { mp.reflective = r; }, "Access Reflective properties")
        .def_property("volume_scattering",
                      // Getter: return a reference to the VolumeScattering struct.
                      [](MaterialProperties &mp) -> MaterialProperties::VolumeScattering &
                      { return mp.volumeScattering; },
                      // Setter: set the volumeScattering field.
                      [](MaterialProperties &mp, const MaterialProperties::VolumeScattering &v)
                      { mp.volumeScattering = v; }, "Access VolumeScattering properties")
        .def("__repr__", []()
             { return "<MaterialProperties>"; });

    // Bind Material.
    py::class_<Material>(m, "Material")
        .def(py::init<>())
        // Constructor for REFRACTIVE material with complex refractive index
        .def(py::init([](MaterialType type, const std::complex<float> &refractive_index)
                      {
            if (type != REFRACTIVE) {
                throw py::value_error("Complex refractive index can only be used with REFRACTIVE material type");
            }
            Material mat;
            mat.type = type;
            mat.properties.refractive.refractiveIndex.r = refractive_index.real();
            mat.properties.refractive.refractiveIndex.i = refractive_index.imag();
            return mat; }),
             py::arg("type"), py::arg("refractive_index"))
        // Constructor for VOLUME_SCATTERING material with complex refractive index, scattering coefficient, and asymetryParameter
        .def(py::init([](MaterialType type, const std::complex<float> &refractive_index, float scattering_coefficient, float asymetryParameter)
                      {
            if (type != VOLUME_SCATTERING) {
                throw py::value_error("Parameters can only be used with VOLUME_SCATTERING material type");
            }
            Material mat;
            mat.type = type;
            mat.properties.volumeScattering.refractiveIndex.r = refractive_index.real();
            mat.properties.volumeScattering.refractiveIndex.i = refractive_index.imag();
            mat.properties.volumeScattering.scatteringCoefficient = scattering_coefficient;
            mat.properties.volumeScattering.asymetryParameter = asymetryParameter;
            return mat; }),
             py::arg("type"), py::arg("refractive_index"), py::arg("scattering_coefficient"), py::arg("g"))
        // Constructor for DIFFUSE material with albedo
        .def(py::init([](MaterialType type, float albedo)
                      {
            if (type != DIFFUSE) {
                throw py::value_error("Albedo parameter can only be used with DIFFUSE material type");
            }
            if (albedo < 0.0f || albedo > 1.0f) {
                throw py::value_error("Albedo must be between 0.0 and 1.0, got " + std::to_string(albedo));
            }
            Material mat;
            mat.type = type;
            mat.properties.diffuse.albedo = albedo;
            return mat; }),
             py::arg("type"), py::arg("albedo"))
        // Constructor for REFLECTIVE material with reflectivity and fuzziness
        .def(py::init([](MaterialType type, float reflectivity, float fuzziness)
                      {
            if (type != REFLECTIVE) {
                throw py::value_error("Reflectivity and fuzziness can only be used with REFLECTIVE material type");
            }
            if (reflectivity < 0.0f || reflectivity > 1.0f) {
                throw py::value_error("Reflectivity must be between in [0.0, 1.0], got " + std::to_string(reflectivity));
            }
            if (fuzziness < 0.0f || fuzziness >= 1.0f) {
                throw py::value_error("Fuzziness must be between in [0.0, 1.0), got " + std::to_string(fuzziness));
            }
            Material mat;
            mat.type = type;
            mat.properties.reflective.reflectivity = reflectivity;
            mat.properties.reflective.fuzziness = fuzziness;
            return mat; }),
             py::arg("type"), py::arg("reflectivity"), py::arg("fuzziness"))
        // Constructor for REFLECTIVE material with reflectivity only (fuzziness=0)
        .def(py::init([](MaterialType type, float reflectivity)
                      {
            if (type != REFLECTIVE) {
                throw py::value_error("Reflectivity can only be used with REFLECTIVE material type");
            }
            Material mat;
            mat.type = type;
            mat.properties.reflective.reflectivity = reflectivity;
            mat.properties.reflective.fuzziness = 0.0f; // Default fuzziness to 0
            return mat; }),
             py::arg("type"), py::arg("reflectivity"))
        // Constructor for REFRACTIVE material with RefractiveIndex object
        .def(py::init([](MaterialType type, const RefractiveIndex &ri)
                      {
            if (type != REFRACTIVE) {
                throw py::value_error("RefractiveIndex can only be used with REFRACTIVE material type");
            }
            Material mat;
            mat.type = type;
            mat.properties.refractive.refractiveIndex = ri;
            return mat; }),
             py::arg("type"), py::arg("refractive_index"))
        .def_readwrite("type", &Material::type)
        .def_readwrite("properties", &Material::properties)
        .def("__repr__", [](const Material &mat)
             {
            std::string type_str = (mat.type == DIFFUSE) ? "DIFFUSE" : "REFRACTIVE";
            if (mat.type == DIFFUSE) {
                return "<Material(DIFFUSE, albedo=" + std::to_string(mat.properties.diffuse.albedo) + ")>";
            } else if (mat.type == REFLECTIVE) {
                return "<Material(REFLECTIVE, reflectivity=" + std::to_string(mat.properties.reflective.reflectivity) + 
                       ", fuzziness=" + std::to_string(mat.properties.reflective.fuzziness) + ")>";
            } else if (mat.type == VOLUME_SCATTERING) {
                return "<Material(VOLUME_SCATTERING, n=" + std::to_string(mat.properties.volumeScattering.refractiveIndex.r) + 
                       "+" + std::to_string(mat.properties.volumeScattering.refractiveIndex.i) + "j, scattering_coefficient=" +
                       std::to_string(mat.properties.volumeScattering.scatteringCoefficient) + ", asymetry_parameter=" +
                       std::to_string(mat.properties.volumeScattering.asymetryParameter) + ")>";
            } else {
                return "<Material(REFRACTIVE, n=" + std::to_string(mat.properties.refractive.refractiveIndex.r) + 
                       "+" + std::to_string(mat.properties.refractive.refractiveIndex.i) + "j)>";
            } });

    // Use std::shared_ptr as the holder type to match Simulation::setGeometry.
    py::class_<IGeometry, std::shared_ptr<IGeometry>>(m, "IGeometry")
        .def("get_type", &IGeometry::getType)
        .def("free_device_memory", &IGeometry::freeDeviceMemory,
             "Free device-side acceleration structure buffers owned by this geometry.");

    py::class_<MeshGeometry, IGeometry, std::shared_ptr<MeshGeometry>>(m, "MeshGeometry")
        .def(py::init([](py::array_t<float> vertices, py::array_t<unsigned int> indices)
                      {
        // Validate vertices array
        if (vertices.ndim() != 2) {
            throw py::value_error("vertices must be a 2D array, got " + 
                                std::to_string(vertices.ndim()) + " dimensions");
        }
        if (vertices.shape(1) != 3) {
            throw py::value_error("vertices must have exactly 3 columns (x, y, z), got " + 
                                std::to_string(vertices.shape(1)) + " columns");
        }
        if (vertices.shape(0) == 0) {
            throw py::value_error("vertices array cannot be empty");
        }
        
        // Validate indices array
        if (indices.ndim() == 1) {
            // 1D array: flattened indices
            if (indices.shape(0) % 3 != 0) {
                throw py::value_error("1D indices array length must be divisible by 3 (triangles), got " + 
                                    std::to_string(indices.shape(0)) + " indices");
            }
        } else if (indices.ndim() == 2) {
            // 2D array: Nx3 indices
            if (indices.shape(1) != 3) {
                throw py::value_error("2D indices array must have exactly 3 columns (triangles), got " + 
                                    std::to_string(indices.shape(1)) + " columns");
            }
        } else {
            throw py::value_error("indices must be a 1D or 2D array, got " + 
                                std::to_string(indices.ndim()) + " dimensions");
        }
        
        // Convert numpy [N,3] array to vector<float3>
        auto vertices_array = vertices.unchecked<2>();
        std::vector<float3> vertex_vec;
        vertex_vec.reserve(vertices_array.shape(0));
        for (py::ssize_t i = 0; i < vertices_array.shape(0); ++i) {
            float3 v;
            v.x = vertices_array(i, 0);
            v.y = vertices_array(i, 1);
            v.z = vertices_array(i, 2);
            vertex_vec.push_back(v);
        }
        
        // Convert indices to vector<unsigned int>
        std::vector<unsigned int> index_vec;
        if (indices.ndim() == 1) {
            // Flattened 1D array
            auto indices_array = indices.unchecked<1>();
            index_vec.reserve(indices_array.shape(0));
            for (py::ssize_t i = 0; i < indices_array.shape(0); ++i) {
                index_vec.push_back(indices_array(i));
            }
        } else {
            // 2D Nx3 array
            auto indices_array = indices.unchecked<2>();
            index_vec.reserve(indices_array.shape(0) * 3);
            for (py::ssize_t i = 0; i < indices_array.shape(0); ++i) {
                for (py::ssize_t j = 0; j < 3; ++j) {
                    index_vec.push_back(indices_array(i, j));
                }
            }
        }
        
        return std::make_shared<MeshGeometry>(std::move(vertex_vec), std::move(index_vec)); }),
             py::arg("vertices"), py::arg("indices"));

    // For InstanceGeometry with numpy arrays
    py::class_<InstanceGeometry, IGeometry, std::shared_ptr<InstanceGeometry>>(m, "InstanceGeometry")
        .def(
            py::init([](std::vector<std::shared_ptr<IGeometry>> sub_geometries,
                        py::array_t<float> instance_transforms,
                        py::array_t<unsigned int> particle_type_ids,
                        py::array_t<unsigned int> material_ids)
                     {
                // Validate sub_geometries
                if (sub_geometries.empty()) {
                    throw py::value_error("sub_geometries cannot be empty");
                }
                
                // Validate instance_transforms array [N,4,3]
                if (instance_transforms.ndim() != 3) {
                    throw py::value_error("instance_transforms must be a 3D array, got " + 
                                        std::to_string(instance_transforms.ndim()) + " dimensions");
                }
                if (instance_transforms.shape(1) != 3) {
                    throw py::value_error("instance_transforms must have 3 rows per transform matrix, got " + 
                                        std::to_string(instance_transforms.shape(1)) + " rows");
                }
                if (instance_transforms.shape(2) != 4) {
                    throw py::value_error("instance_transforms must have 4 columns per transform matrix, got " + 
                                        std::to_string(instance_transforms.shape(2)) + " columns");
                }
                if (instance_transforms.shape(0) == 0) {
                    throw py::value_error("instance_transforms array cannot be empty");
                }
                
                // Validate particle_type_ids array [N]
                if (particle_type_ids.ndim() != 1) {
                    throw py::value_error("particle_type_ids must be a 1D array, got " + 
                                        std::to_string(particle_type_ids.ndim()) + " dimensions");
                }
                if (particle_type_ids.shape(0) != instance_transforms.shape(0)) {
                    throw py::value_error("particle_type_ids length (" + 
                                        std::to_string(particle_type_ids.shape(0)) + 
                                        ") must match number of instances (" + 
                                        std::to_string(instance_transforms.shape(0)) + ")");
                }
                
                // Validate material_ids array [N]
                if (material_ids.ndim() != 1) {
                    throw py::value_error("material_ids must be a 1D array, got " + 
                                        std::to_string(material_ids.ndim()) + " dimensions");
                }
                if (material_ids.shape(0) != instance_transforms.shape(0)) {
                    throw py::value_error("material_ids length (" + 
                                        std::to_string(material_ids.shape(0)) + 
                                        ") must match number of instances (" + 
                                        std::to_string(instance_transforms.shape(0)) + ")");
                }
                
                // Convert numpy [N,3,4] array to vector<float>
                auto transforms_array = instance_transforms.unchecked<3>();
                std::vector<float> transform_vec;
                transform_vec.reserve(transforms_array.shape(0) * 12); // 3,4 = 12 floats per transform
                for (py::ssize_t i = 0; i < transforms_array.shape(0); ++i) {
                    for (py::ssize_t j = 0; j < 3; ++j) {
                        for (py::ssize_t k = 0; k < 4; ++k) {
                            transform_vec.push_back(transforms_array(i, j, k));
                        }
                    }
                }
                
                // Convert numpy [N] arrays to vector<unsigned int>
                auto type_ids_array = particle_type_ids.unchecked<1>();
                std::vector<unsigned int> type_ids_vec;
                type_ids_vec.reserve(type_ids_array.shape(0));
                for (py::ssize_t i = 0; i < type_ids_array.shape(0); ++i) {
                    type_ids_vec.push_back(type_ids_array(i));
                }

                auto material_ids_array = material_ids.unchecked<1>();
                std::vector<unsigned int> material_ids_vec;
                material_ids_vec.reserve(material_ids_array.shape(0));
                for (py::ssize_t i = 0; i < material_ids_array.shape(0); ++i) {
                    material_ids_vec.push_back(material_ids_array(i));
                }
                
                return std::make_shared<InstanceGeometry>(std::move(sub_geometries),
                                                          std::move(transform_vec),
                                                          std::move(type_ids_vec),
                                                          std::move(material_ids_vec)); }),
            py::arg("sub_geometries"),
            py::arg("instance_transforms"),
            py::arg("particle_type_ids"),
            py::arg("material_ids"));

    py::class_<IRayGenerator, std::shared_ptr<IRayGenerator>>(m, "IRayGenerator");

    py::class_<ParallelRayGenerator, IRayGenerator, std::shared_ptr<ParallelRayGenerator>>(m, "ParallelRayGenerator")
        .def(py::init([](int number_of_rays, py::object origin, py::object direction, float offset_radius)
                      { return ParallelRayGenerator(
                            number_of_rays,
                            toFloat3(origin, "origin"),
                            toFloat3(direction, "direction"),
                            offset_radius); }),
             py::arg("number_of_rays"), py::arg("origin"), py::arg("direction"), py::arg("offset_radius"))
        .def_property("origin", [](const ParallelRayGenerator &self)
                      { return toPyObject(self.getOrigin()); }, [](ParallelRayGenerator &self, py::object origin)
                      { self.setOrigin(toFloat3(origin, "origin")); })
        .def_property("direction", [](const ParallelRayGenerator &self)
                      { return toPyObject(self.getDirection()); }, [](ParallelRayGenerator &self, py::object direction)
                      { self.setDirection(toFloat3(direction, "direction")); })
        .def_property("offset_radius", &ParallelRayGenerator::getOffsetRadius, &ParallelRayGenerator::setOffsetRadius);

    py::class_<IsotropicRayGenerator, IRayGenerator, std::shared_ptr<IsotropicRayGenerator>>(m, "IsotropicRayGenerator")
        .def(py::init([](int number_of_rays, py::object center, float source_radius, float offset_radius)
                      { return IsotropicRayGenerator(
                            number_of_rays,
                            toFloat3(center, "center"),
                            source_radius,
                            offset_radius); }),
             py::arg("number_of_rays"), py::arg("center"), py::arg("source_radius"), py::arg("offset_radius"))
        .def_property("center", [](const IsotropicRayGenerator &self)
                      { return toPyObject(self.getCenter()); }, [](IsotropicRayGenerator &self, py::object center)
                      { self.setCenter(toFloat3(center, "center")); })
        .def_property("source_radius", &IsotropicRayGenerator::getSourceRadius, &IsotropicRayGenerator::setSourceRadius)
        .def_property("offset_radius", &IsotropicRayGenerator::getOffsetRadius, &IsotropicRayGenerator::setOffsetRadius);

    py::class_<CameraRayGenerator, IRayGenerator, std::shared_ptr<CameraRayGenerator>>(m, "CameraRayGenerator")
        .def(py::init<>())
        .def_property("image_width", &CameraRayGenerator::getImageWidth, &CameraRayGenerator::setImageWidth)
        .def_property("image_height", &CameraRayGenerator::getImageHeight, &CameraRayGenerator::setImageHeight)
        .def_property("aspect_ratio", &CameraRayGenerator::getAspectRatio, &CameraRayGenerator::setAspectRatio)
        .def_property("samples_per_pixel", &CameraRayGenerator::getSamplesPerPixel, &CameraRayGenerator::setSamplesPerPixel)
        .def_property("vertical_fov", &CameraRayGenerator::getVerticalFov, &CameraRayGenerator::setVerticalFov)
        .def_property("look_from", [](const CameraRayGenerator &self)
                      { return toPyObject(self.getLookFrom()); }, [](CameraRayGenerator &self, py::object value)
                      { self.setLookFrom(toFloat3(value, "look_from")); })
        .def_property("look_at", [](const CameraRayGenerator &self)
                      { return toPyObject(self.getLookAt()); }, [](CameraRayGenerator &self, py::object value)
                      { self.setLookAt(toFloat3(value, "look_at")); })
        .def_property("vertical_up", [](const CameraRayGenerator &self)
                      { return toPyObject(self.getVerticalUp()); }, [](CameraRayGenerator &self, py::object value)
                      { self.setVerticalUp(toFloat3(value, "vertical_up")); })
        .def_property("defocus_angle", &CameraRayGenerator::getDefocusAngle, &CameraRayGenerator::setDefocusAngle)
        .def_property("focus_distance", &CameraRayGenerator::getFocusDistance, &CameraRayGenerator::setFocusDistance);

    py::class_<Simulation, std::shared_ptr<Simulation>>(m, "Simulation")
        .def(py::init<int, int, bool>(), py::arg("gpu_id") = 0, py::arg("optix_logging_level") = 1, py::arg("enable_validation_mode") = false)
        .def("run", &Simulation::run, "Run the simulation")
        .def("free_device_memory", &Simulation::freeDeviceMemory,
             "Free device-side buffers held by the simulation (pipeline, outputs, and geometry acceleration structures).")
        .def(
            "calculate_volume_fraction",
            [](Simulation &sim, py::object boxMin, py::object boxMax, uint32_t numSamples)
            {
                return sim.calculateVolumeFraction(
                    toFloat3(boxMin, "box_min"),
                    toFloat3(boxMax, "box_max"),
                    numSamples);
            },
            py::arg("box_min"),
            py::arg("box_max"),
            py::arg("number_of_samples"),
            "Use Monte-Carlo sampling to calculate the volume fraction in a given box")
        .def("get_output_buffer", [](Simulation &sim, OutputType type) -> py::array
             {
            auto *deviceBuffers = sim.getDeviceOutputBuffers();
            if (!deviceBuffers)
            {
                throw std::runtime_error("Output buffers not initialized. Call run() first.");
            }

            auto rayOutput = sim.getRayOutput();
            if (!rayOutput)
            {
                throw std::runtime_error("Outputs not configured. Call configure_outputs() first.");
            }

            const auto &descriptor = rayOutput->getDescriptor(type);
            const BufferLayout &layout = deviceBuffers->getBufferLayout(type);
            if (!layout.devicePtr)
            {
                throw std::runtime_error("Output buffer for the specified type is not available.");
            }

            const uint3 shape = layout.shape;
            const size_t elementCount = static_cast<size_t>(shape.x) * shape.y * shape.z;
            const size_t totalBytes = elementCount * descriptor.elementSize;

            auto baseDimensions = [&]() -> std::vector<py::ssize_t>
            {
                std::vector<py::ssize_t> dims;
                auto pushDim = [&](unsigned int value, bool force)
                {
                    if (value > 1 || force)
                    {
                        dims.push_back(static_cast<py::ssize_t>(value));
                    }
                };
                pushDim(shape.z, false);
                pushDim(shape.y, false);
                pushDim(shape.x, dims.empty());
                if (dims.empty())
                {
                    dims.push_back(1);
                }
                return dims;
            };

            auto toShapeContainer = [](const std::vector<py::ssize_t> &dims) -> py::array::ShapeContainer
            {
                return py::array::ShapeContainer(dims.begin(), dims.end());
            };

            auto shapeFor = [&](py::ssize_t components) -> py::array::ShapeContainer
            {
                auto dims = baseDimensions();
                if (components > 1)
                {
                    dims.push_back(components);
                }
                return toShapeContainer(dims);
            };

            auto copyFromDevice = [&](void *hostPtr)
            {
                if (totalBytes == 0)
                {
                    return;
                }
                const cudaError_t err = cudaMemcpy(hostPtr, layout.devicePtr, totalBytes, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    const std::string errorMsg = std::string("Failed to copy output buffer '") + descriptor.name +
                                                 "' from device memory: " + cudaGetErrorString(err);
                    throw std::runtime_error(errorMsg);
                }
            };

            switch (descriptor.elementType)
            {
            case BufferDescriptor::ElementType::Float3:
            {
                py::array_t<float> result(shapeFor(3));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::Float4:
            {
                py::array_t<float> result(shapeFor(4));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::Float:
            {
                py::array_t<float> result(shapeFor(1));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::Double:
            {
                py::array_t<double> result(shapeFor(1));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::Int32:
            {
                py::array_t<int32_t> result(shapeFor(1));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::UInt32:
            {
                py::array_t<uint32_t> result(shapeFor(1));
                copyFromDevice(result.mutable_data());
                return result;
            }
            case BufferDescriptor::ElementType::String:
            {
                py::array_t<char> result(shapeFor(static_cast<py::ssize_t>(LOG_BYTES_PER_RAY)));
                copyFromDevice(result.mutable_data());
                return result;
            }
            default:
                throw std::runtime_error("Unsupported buffer element type.");
            } }, py::arg("output_type"), "Get the specified output buffer as a numpy array")

        .def_property("geometry", &Simulation::getGeometry, &Simulation::setGeometry, "Get or set the geometry for the simulation.")
        .def_property("materials", &Simulation::getMaterials, &Simulation::setMaterials, "Get or set the materials for the simulation.")
        .def_property("wavelength_um", &Simulation::getWavelengthUm, &Simulation::setWavelengthUm, "Get or set the wavelength in micrometers for the simulation.")
        .def_property("length_unit", &Simulation::getLengthUnit, &Simulation::setLengthUnit, "Get or set the geometry length unit for the simulation.")
        .def_property("stokes_vector", [](const Simulation &sim)
                      { return toPyObject(sim.getStokesVector()); }, [](Simulation &sim, py::object vec)
                      { sim.setStokesVector(toFloat4(vec, "stokes_vector")); }, "Get or set the initial Stokes vector for polarization.")
        .def_property("q_minus_axis_seed", [](const Simulation &sim)
                      { return toPyObject(sim.getQMinusAxisSeed()); }, [](Simulation &sim, py::object vec)
                      { sim.setQMinusAxisSeed(toFloat3(vec, "q_minus_axis_seed")); }, "Get or set the Q- axis for the source ray.")
        .def_property("max_scattering_count", &Simulation::getMaxScatteringCount, &Simulation::setMaxScatteringCount, "Get or set the maximum scattering count for ray tracing. If 0, there is no limit.")
        .def_property("max_nested_geometry_levels", &Simulation::getMaxNestedGeometryLevels, &Simulation::setMaxNestedGeometryLevels, "Get or set the maximum nested geometry levels for ray tracing.")
        .def_property_readonly("max_sub_geometries", &Simulation::getMaxSubGeometries, "Get the maximum number of sub-geometries allowed in an InstanceGeometry.")
        .def_property_readonly("max_mesh_triangles", &Simulation::getMaxMeshTriangles, "Get the maximum number of triangles allowed in a MeshGeometry.")
        .def_property("seed", &Simulation::getInitSeed, &Simulation::setInitSeed, "Get or set the initial seed for random number generation.")
        .def_property("use_complex_fresnel", &Simulation::getUseComplexFresnel, &Simulation::setUseComplexFresnel, "Get or set whether to use complex Fresnel equations for refractive materials.")
        .def_property("direction_healpix_nside", &Simulation::getDirectionHealpixNside, &Simulation::setDirectionHealpixNside, "Set NSIDE for aggregating miss directions into Healpix bins (0 disables the histogram).")
        .def_property("ray_generator", &Simulation::getRayGenerator, &Simulation::setRayGenerator, "Get or set the ray generator for the simulation.")
        .def_property("outputs", [](const Simulation &sim) -> std::vector<OutputType>
                      {
            if (!sim.getRayOutput()) {
                throw std::runtime_error("Outputs not configured. Call configure_outputs() first.");
            }

            const auto &descriptors = sim.getRayOutput()->getDescriptors();
            std::vector<OutputType> outputTypes;
            for (const auto &desc : descriptors) {
                if (desc.enabled) {
                    outputTypes.push_back(desc.type);
                }
            }
            return outputTypes; }, [](Simulation &sim, const std::vector<OutputType> &outputTypes)
                      {
                          auto rayTracingOutput = std::make_shared<RayTracingOutput>();
                          for (const auto &type : outputTypes)
                          {
                              rayTracingOutput->enableOutput(type, true); // Enable the buffer
                          }
                          sim.setRayOutput(rayTracingOutput); // Set the configured output buffers in the simulation
                      },
                      "Get or set the list of output types for the simulation.");

    // Module-level function to check CUDA availability
    m.def("is_cuda_available", []() -> bool
          {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        return err == cudaSuccess && deviceCount > 0; }, "Check if CUDA is available and properly initialized. Returns True if CUDA device(s) are detected.");
}