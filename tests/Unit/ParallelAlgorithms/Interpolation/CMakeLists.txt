# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelInterpolation")

set(LIBRARY_SOURCES
  Test_AddTemporalIdsToInterpolationTarget.cpp
  Test_CleanUpInterpolator.cpp
  Test_ElementReceiveInterpPoints.cpp
  Test_InitializeInterpolationTarget.cpp
  Test_InitializeInterpolator.cpp
  Test_Interpolate.cpp
  Test_InterpolateWithoutInterpComponent.cpp
  Test_InterpolationTargetApparentHorizon.cpp
  Test_InterpolationTargetKerrHorizon.cpp
  Test_InterpolationTargetLineSegment.cpp
  Test_InterpolationTargetReceiveVars.cpp
  Test_InterpolationTargetVarsFromElement.cpp
  Test_InterpolationTargetSpecifiedPoints.cpp
  Test_InterpolationTargetSphere.cpp
  Test_InterpolationTargetWedgeSectionTorus.cpp
  Test_InterpolatorReceivePoints.cpp
  Test_InterpolatorReceiveVolumeData.cpp
  Test_InterpolatorRegisterElement.cpp
  Test_ObserveTimeSeriesAndSurfaceData.cpp
  Test_ParallelInterpolator.cpp
  Test_Protocols.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/Interpolation/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  CoordinateMaps
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  GeneralRelativitySolutions
  Interpolation
  IO
  IoTestHelpers
  Logging
  MathFunctions
  ParallelInterpolation
  RelativisticEulerSolutions
  Spectral
  Utilities
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
