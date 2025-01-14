# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DgSubcell)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ActiveGrid.hpp
  CartesianFluxDivergence.hpp
  ComputeBoundaryTerms.hpp
  CorrectPackagedData.hpp
  DgSubcell.hpp
  GhostZoneLogicalCoordinates.hpp
  Matrices.hpp
  Mesh.hpp
  NeighborReconstructedFaceSolution.hpp
  PerssonTci.hpp
  PrepareNeighborData.hpp
  Projection.hpp
  RdmpTci.hpp
  RdmpTciData.hpp
  Reconstruction.hpp
  ReconstructionMethod.hpp
  SliceData.hpp
  SliceTensor.hpp
  SliceVariable.hpp
  SubcellOptions.hpp
  TciStatus.hpp
  TwoMeshRdmpTci.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ActiveGrid.cpp
  CartesianFluxDivergence.cpp
  GhostZoneLogicalCoordinates.cpp
  Matrices.cpp
  Mesh.cpp
  PerssonTci.cpp
  Projection.cpp
  RdmpTci.cpp
  RdmpTciData.cpp
  Reconstruction.cpp
  ReconstructionMethod.cpp
  SliceData.cpp
  SubcellOptions.cpp
  TciStatus.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  IO
  Spectral

  INTERFACE
  Events
  FunctionsOfTime
  Parallel
  Time

  PRIVATE
  Blas
  ErrorHandling
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(Tags)
