// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Cce {
namespace Actions {
/// \ingroup ActionsGroup
/// \brief Interpolates and sends points to the `CceWorldtubeTarget`
///
/// This is invoked on DgElementArray for the GH system.
/// This action should appear only in the self-start action list, and
/// is an unfortunate hack needed by the contradictory constraints
/// of a locally-stepped CCE system and the events and dense triggers
/// during the self start procedure.
template <typename CceWorltubeTargetTag>
struct InterpolateDuringSelfStart {
  template <typename DbTags, typename Metavariables, typename... InboxTags,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const component) {
    auto interpolate_event = intrp::Events::InterpolateWithoutInterpComponent<
        Metavariables::volume_dim, CceWorltubeTargetTag, Metavariables,
        typename CceWorltubeTargetTag::vars_to_interpolate_to_target>{};
    interpolate_event.run(make_observation_box<db::AddComputeTags<>>(box),
                          cache, array_index, component);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Cce
