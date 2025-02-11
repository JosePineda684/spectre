// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"

namespace evolution::dg::subcell {
namespace {
void test_impl(const std::vector<double>& expected_values,
               const size_t incorrect_value_index) {
  std::vector<double> values = expected_values;
  values[incorrect_value_index] += 0.1;
  const fd::ReconstructionMethod recons_method =
      fd::ReconstructionMethod::AllDimsAtOnce;

  CHECK(SubcellOptions(expected_values[0], expected_values[1],
                       expected_values[2], expected_values[3],
                       expected_values[4], expected_values[5], false,
                       recons_method) !=
        SubcellOptions(values[0], values[1], values[2], values[3], values[4],
                       values[5], false, recons_method));
  CHECK_FALSE(SubcellOptions(expected_values[0], expected_values[1],
                             expected_values[2], expected_values[3],
                             expected_values[4], expected_values[5], false,
                             recons_method) ==
              SubcellOptions(values[0], values[1], values[2], values[3],
                             values[4], values[5], false, recons_method));

  CHECK(
      SubcellOptions(expected_values[0], expected_values[1], expected_values[2],
                     expected_values[3], expected_values[4], expected_values[5],
                     false, recons_method) !=
      SubcellOptions(expected_values[0], expected_values[1], expected_values[2],
                     expected_values[3], expected_values[4], expected_values[5],
                     true, recons_method));
  CHECK_FALSE(
      SubcellOptions(expected_values[0], expected_values[1], expected_values[2],
                     expected_values[3], expected_values[4], expected_values[5],
                     false, recons_method) ==
      SubcellOptions(expected_values[0], expected_values[1], expected_values[2],
                     expected_values[3], expected_values[4], expected_values[5],
                     true, recons_method));

  CHECK(
      SubcellOptions(expected_values[0], expected_values[1], expected_values[2],
                     expected_values[3], expected_values[4], expected_values[5],
                     false, recons_method) !=
      SubcellOptions(expected_values[0], expected_values[1], expected_values[2],
                     expected_values[3], expected_values[4], expected_values[5],
                     false, fd::ReconstructionMethod::DimByDim));
  CHECK_FALSE(
      SubcellOptions(expected_values[0], expected_values[1], expected_values[2],
                     expected_values[3], expected_values[4], expected_values[5],
                     false, recons_method) ==
      SubcellOptions(expected_values[0], expected_values[1], expected_values[2],
                     expected_values[3], expected_values[4], expected_values[5],
                     false, fd::ReconstructionMethod::DimByDim));
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.SubcellOptions",
                  "[Evolution][Unit]") {
  const std::vector<double> expected_values{1.0e-3, 1.0e-4, 2.0e-3,
                                            2.0e-4, 5.0,    4.0};
  for (size_t i = 0; i < expected_values.size(); ++i) {
    test_impl(expected_values, i);
  }

  const SubcellOptions options(expected_values[0], expected_values[1],
                               expected_values[2], expected_values[3],
                               expected_values[4], expected_values[5], true,
                               fd::ReconstructionMethod::DimByDim);
  const SubcellOptions deserialized_options =
      serialize_and_deserialize(options);
  CHECK(options == deserialized_options);

  CHECK(options == TestHelpers::test_option_tag<OptionTags::SubcellOptions>(
                       "InitialData:\n"
                       "  RdmpDelta0: 1.0e-3\n"
                       "  RdmpEpsilon: 1.0e-4\n"
                       "  PerssonExponent: 5.0\n"
                       "RdmpDelta0: 2.0e-3\n"
                       "RdmpEpsilon: 2.0e-4\n"
                       "PerssonExponent: 4.0\n"
                       "AlwaysUseSubcells: true\n"
                       "SubcellToDgReconstructionMethod: DimByDim\n"));
}
}  // namespace
}  // namespace evolution::dg::subcell
