// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/Cerk2.hpp"
#include "Time/TimeSteppers/Cerk3.hpp"
#include "Time/TimeSteppers/Cerk4.hpp"
#include "Time/TimeSteppers/Cerk5.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/RungeKutta4.hpp"
#include "Utilities/TMPL.hpp"

namespace TimeSteppers {
/// Typelist of available TimeSteppers
using time_steppers =
    tmpl::list<TimeSteppers::AdamsBashforthN, TimeSteppers::Cerk2,
               TimeSteppers::Cerk3, TimeSteppers::Cerk4, TimeSteppers::Cerk5,
               TimeSteppers::DormandPrince5, TimeSteppers::RungeKutta3,
               TimeSteppers::RungeKutta4>;

/// Typelist of available LtsTimeSteppers
using lts_time_steppers = tmpl::list<TimeSteppers::AdamsBashforthN>;
}  // namespace Triggers
