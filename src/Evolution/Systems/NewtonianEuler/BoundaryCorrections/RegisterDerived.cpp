// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/Factory.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace NewtonianEuler::BoundaryCorrections {
void register_derived_with_charm() {
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<1>>();
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<2>>();
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<3>>();
}
}  // namespace NewtonianEuler::BoundaryCorrections
