/**
 * @file start_blend.hpp
 * @brief Eases the joint targets in when the policy takes over the robot.
 *
 * While the program waits for the operator's start signal, the robot is held
 * at the default pose. The policy's first targets are generally not that
 * pose: with soft gains a policy asks for a target some way past where a
 * joint is, to get the torque it wants. Sending those targets as they are
 * steps every joint target in a single control tick, and the robot jolts.
 *
 * StartBlend replaces that step with a blend, over a chosen time, from the
 * targets that were being held into the policy's targets. A duration of zero
 * leaves the policy's targets untouched.
 */

#ifndef START_BLEND_HPP
#define START_BLEND_HPP

#include <algorithm>
#include <array>
#include <cstddef>

/**
 * @brief Blends N joint targets from a held pose into the policy's targets.
 *
 * Call Begin() once, when control starts, with the targets held until then.
 * Call Apply() on every control tick with the policy's targets; it changes
 * them in place while the blend lasts and leaves them alone afterwards.
 */
template <std::size_t N>
class StartBlend {
 public:
  /// @param duration_seconds How long the blend lasts. Zero or less: no blend.
  explicit StartBlend(double duration_seconds = 0.0)
      : duration_(std::max(0.0, duration_seconds)), elapsed_(duration_) {}

  /// Start blending away from `held_targets`, the targets in force until now.
  void Begin(const std::array<float, N>& held_targets) {
    from_ = held_targets;
    elapsed_ = 0.0;
  }

  /// Advance the blend by `dt` seconds and apply it to `targets` in place.
  void Apply(double dt, std::array<float, N>& targets) {
    if (elapsed_ >= duration_) {
      return;
    }
    elapsed_ = std::min(elapsed_ + dt, duration_);
    const double weight = Weight();
    for (std::size_t i = 0; i < N; ++i) {
      targets[i] = static_cast<float>(from_[i] * (1.0 - weight) + targets[i] * weight);
    }
  }

  /// Share of the policy's targets in use: 0 at Begin(), 1 once the blend is over.
  double Weight() const {
    if (elapsed_ >= duration_) {
      return 1.0;
    }
    // Smoothstep: the weight and its rate of change both start and end at
    // zero, so neither the targets nor their speed jump.
    const double u = elapsed_ / duration_;
    return u * u * (3.0 - 2.0 * u);
  }

  /// True from Begin() until the blend is over.
  bool Active() const { return elapsed_ < duration_; }

  double Duration() const { return duration_; }

 private:
  double duration_;  ///< Length of the blend (s).
  double elapsed_;   ///< Time since Begin() (s); equal to duration_ when idle.
  std::array<float, N> from_ = {};  ///< Targets held when the blend began.
};

#endif  // START_BLEND_HPP
