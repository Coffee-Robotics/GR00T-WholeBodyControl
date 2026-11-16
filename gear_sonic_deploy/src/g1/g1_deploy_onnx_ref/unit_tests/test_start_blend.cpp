#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "../include/start_blend.hpp"

namespace {

constexpr double kDt = 0.02;  // the control loop's period
using Targets = std::array<float, 2>;

const Targets kHeld = {0.5f, -1.0f};
const Targets kPolicy = {1.5f, 1.0f};

/// Run the blend for `ticks` control ticks against a constant policy output
/// and return the targets sent on the last of them.
Targets RunTicks(StartBlend<2>& blend, int ticks) {
  Targets sent = kPolicy;
  for (int i = 0; i < ticks; ++i) {
    sent = kPolicy;
    blend.Apply(kDt, sent);
  }
  return sent;
}

TEST(StartBlendTest, ZeroDurationLeavesThePolicyTargetsAlone) {
  StartBlend<2> blend(0.0);
  blend.Begin(kHeld);
  EXPECT_FALSE(blend.Active());

  Targets sent = kPolicy;
  blend.Apply(kDt, sent);
  EXPECT_EQ(sent, kPolicy);
}

TEST(StartBlendTest, NegativeDurationIsTreatedAsNoBlend) {
  StartBlend<2> blend(-1.0);
  blend.Begin(kHeld);

  Targets sent = kPolicy;
  blend.Apply(kDt, sent);
  EXPECT_EQ(sent, kPolicy);
}

TEST(StartBlendTest, DoesNothingBeforeBegin) {
  StartBlend<2> blend(1.0);
  EXPECT_FALSE(blend.Active());

  Targets sent = kPolicy;
  blend.Apply(kDt, sent);
  EXPECT_EQ(sent, kPolicy);
}

TEST(StartBlendTest, FirstTickStaysCloseToTheHeldTargets) {
  StartBlend<2> blend(1.0);
  blend.Begin(kHeld);

  const Targets sent = RunTicks(blend, 1);
  // Without the blend the first tick would move each target by the whole
  // gap between held and policy (1 and 2 rad here).
  EXPECT_NEAR(sent[0], kHeld[0], 0.002);
  EXPECT_NEAR(sent[1], kHeld[1], 0.004);
}

TEST(StartBlendTest, HalfwayIsTheMidpoint) {
  StartBlend<2> blend(1.0);
  blend.Begin(kHeld);

  const Targets sent = RunTicks(blend, 25);
  EXPECT_NEAR(sent[0], 1.0, 1e-5);
  EXPECT_NEAR(sent[1], 0.0, 1e-5);
}

TEST(StartBlendTest, EndsOnThePolicyTargetsAndStaysThere) {
  StartBlend<2> blend(1.0);
  blend.Begin(kHeld);

  // One tick more than the duration, in case the summed ticks fall a hair short of it.
  EXPECT_EQ(RunTicks(blend, 51), kPolicy);
  EXPECT_FALSE(blend.Active());
  EXPECT_EQ(RunTicks(blend, 10), kPolicy);
}

TEST(StartBlendTest, NoTickMovesATargetFar) {
  StartBlend<2> blend(1.0);
  blend.Begin(kHeld);

  // The steepest part of a smoothstep moves 1.5 * gap * dt / duration per
  // tick: 0.06 rad for the 2 rad gap of the second joint.
  Targets previous = kHeld;
  for (int i = 0; i < 60; ++i) {
    Targets sent = kPolicy;
    blend.Apply(kDt, sent);
    EXPECT_LE(std::abs(sent[0] - previous[0]), 0.0301) << "tick " << i;
    EXPECT_LE(std::abs(sent[1] - previous[1]), 0.0601) << "tick " << i;
    previous = sent;
  }
  EXPECT_EQ(previous, kPolicy);
}

TEST(StartBlendTest, FollowsAPolicyOutputThatChangesDuringTheBlend) {
  StartBlend<2> blend(1.0);
  blend.Begin(kHeld);
  RunTicks(blend, 24);

  // Halfway through, the blend is half held pose and half whatever the
  // policy asks for on that tick.
  Targets sent = {2.5f, 3.0f};
  blend.Apply(kDt, sent);
  EXPECT_NEAR(sent[0], 0.5 * kHeld[0] + 0.5 * 2.5, 1e-5);
  EXPECT_NEAR(sent[1], 0.5 * kHeld[1] + 0.5 * 3.0, 1e-5);
}

TEST(StartBlendTest, BeginAgainStartsOver) {
  StartBlend<2> blend(1.0);
  blend.Begin(kHeld);
  RunTicks(blend, 60);
  ASSERT_FALSE(blend.Active());

  blend.Begin(kPolicy);
  EXPECT_TRUE(blend.Active());
  Targets sent = kHeld;
  blend.Apply(kDt, sent);
  EXPECT_NEAR(sent[0], kPolicy[0], 0.002);
  EXPECT_NEAR(sent[1], kPolicy[1], 0.004);
}

}  // namespace
