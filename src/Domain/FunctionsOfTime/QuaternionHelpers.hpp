// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for converting between `DataVector`s and boost
/// quaternions.

#pragma once

#include <boost/math/quaternion.hpp>

#include "DataStructures/DataVector.hpp"

/// \cond
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

/// Convert a `boost::math::quaternion` to a `DataVector`
DataVector quaternion_to_datavector(
    const boost::math::quaternion<double>& input) noexcept;

/// \brief Convert a `DataVector` to a `boost::math::quaternion`
///
/// \details To convert to a quaternion, a `DataVector` must have either 3 or 4
/// components. If it has 3 components, the quaternion will be constructed with
/// 0 scalar part while the vector part is the `DataVector`. If the `DataVector`
/// has 4 components, the quaternion is just the `DataVector` itself.
boost::math::quaternion<double> datavector_to_quaternion(
    const DataVector& input) noexcept;

/// Normalize a `boost::math::quaternion`
void normalize_quaternion(
    gsl::not_null<boost::math::quaternion<double>*> input) noexcept;
