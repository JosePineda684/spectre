# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FiniteDifference)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AoWeno.cpp
  FallbackReconstructorType.cpp
  Minmod.cpp
  MonotonisedCentral.cpp
  Unlimited.cpp
  Wcns5z.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AoWeno.hpp
  FallbackReconstructorType.hpp
  FiniteDifference.hpp
  Minmod.hpp
  MonotonisedCentral.hpp
  Reconstruct.hpp
  Reconstruct.tpp
  Unlimited.hpp
  Wcns5z.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Utilities
  PRIVATE
  DataStructures
  DomainStructure
  ErrorHandling
  )
