package rl.agent

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import org.scalatest.funsuite.AnyFunSuite
import rl.env.GridWorld1D
import rl.logging.NoOpLogger
import rl.TestFixture.AgentTestFixture

class TestSarsa extends AnyFunSuite {

  def createSarsa(
    fixture: AgentTestFixture,
    nSteps: Int = 1,
  ): Sarsa[GridWorld1D] = new Sarsa[GridWorld1D](
    env = fixture.env,
    qTable = fixture.qTable,
    buffer = fixture.buffer,
    nSteps = nSteps,
    learningRate = fixture.learningRate,
    discountFactor = fixture.discountFactor,
    explorationActor = fixture.explorationActor,
    logger = fixture.logger
  )

  test("getNextQValue returns Q-value for the greedy action") {
    val nextState = 3
    val testIO = for {
      fixture <- AgentTestFixture(
        initialQValues = Map((nextState, -1) -> 0, (nextState, 1) -> 1.0)
      )
      sarsa = createSarsa(fixture)

      qValues <- fixture.qTable.get
      actionSpace <- fixture.env.getActionSpace
      nextQValue <- sarsa.getNextQValue(
        nextState = nextState,
        done = false,
        qValues = qValues,
        actionSpace = actionSpace
      )
    } yield nextQValue

    val result = testIO.unsafeRunSync()
    assert(result == 1.0)
  }

  test("getNextQValue returns 0.0 for unknown state-action pairs") {
    val testIO = for {
      fixture <- AgentTestFixture()
      sarsa = createSarsa(fixture)

      qValues <- fixture.qTable.get
      actionSpace <- fixture.env.getActionSpace
      nextQValue <- sarsa.getNextQValue(
        nextState = 2,
        done = false,
        qValues = qValues,
        actionSpace = actionSpace
      )
    } yield nextQValue

    val result = testIO.unsafeRunSync()
    assert(result == 0.0)
  }
}
