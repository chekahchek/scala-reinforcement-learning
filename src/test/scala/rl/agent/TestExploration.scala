package rl.agent

import cats.effect.{IO, Ref}
import cats.effect.unsafe.implicits.global
import org.scalatest.funsuite.AnyFunSuite
import rl.env.GridWorld1D
import rl.logging.NoOpLogger
import rl.TestFixture.AgentTestFixture

class TestExploration extends AnyFunSuite {

  // Common test variables
  val state = 2
  val environment: IO[GridWorld1D] = GridWorld1D(NoOpLogger)
  val actionSpace: IO[List[Int]] = environment.flatMap(_.getActionSpace)
  val qValues: Map[(Int, Int), Double] = Map((state, -1) -> 0.0, (state, 1) -> 1.0)

  test("Epsilon greedy exploits by selecting the best action") {
    val action = for {
      epsilonGreedyActor <- EpsilonGreedy[GridWorld1D](0.0)
      as <- actionSpace
      action <- epsilonGreedyActor.getAction(as, state, qValues)
    } yield action
    val result = action.unsafeRunSync()
    assert(result == 1)
  }

  test("UCB exploits by selecting the action with the highest UCB value") {
    val action = for {
      stateActionCount <- Ref[IO].of(Map((state, -1) -> 1, (state, 1) -> 1))
      ucbActor <- IO.pure(UCB[GridWorld1D](1, stateActionCount))
      as <- actionSpace
      action <- ucbActor.getAction(as, state, qValues)
    } yield action
    val result = action.unsafeRunSync()
    assert(result == 1)
  }

  test("UCB exploits explores when the action has not been selected before") {
    val action = for {
      stateActionCount <- Ref[IO].of(Map((state, -1) -> 0, (state, 1) -> 1000))
      ucbActor <- IO.pure(UCB[GridWorld1D](1, stateActionCount))
      as <- actionSpace
      action <- ucbActor.getAction(as, state, qValues)
    } yield action
    val result = action.unsafeRunSync()
    assert(result == -1)
  }

}
