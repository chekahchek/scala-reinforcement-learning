package rl.agent

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import org.scalatest.funsuite.AnyFunSuite
import rl.env.GridWorld1D
import rl.logging.NoOpLogger
import rl.TestFixture.AgentTestFixture

class TestQLearning extends AnyFunSuite {

  def createQLearning(
    fixture: AgentTestFixture,
    nSteps: Int = 1,
  ): QLearning[GridWorld1D] = new QLearning[GridWorld1D](
    env = fixture.env,
    qTable = fixture.qTable,
    buffer = fixture.buffer,
    nSteps = nSteps,
    learningRate = fixture.learningRate,
    discountFactor = fixture.discountFactor,
    explorationActor = fixture.explorationActor,
    logger = fixture.logger
  )

  test("getNextQValue returns max Q-value across all actions") {
    val nextState = 3
    val testIO = for {
      fixture <- AgentTestFixture(
        initialQValues = Map((nextState, -1) -> 0.5, (nextState, 1) -> 1.0)
      )
      qLearning = createQLearning(fixture)

      qValues <- fixture.qTable.get
      actionSpace <- fixture.env.getActionSpace
      nextQValue <- qLearning.getNextQValue(
        nextState = nextState,
        done = false,
        qValues = qValues,
        actionSpace = actionSpace
      )
    } yield nextQValue

    val result = testIO.unsafeRunSync()
    assert(result == 1.0)
  }

  test("getNextQValue returns 0.0 when done is true") {
    val nextState = 3
    val testIO = for {
      fixture <- AgentTestFixture(
        initialQValues = Map((nextState, -1) -> 0.5, (nextState, 1) -> 1.0)
      )
      qLearning = createQLearning(fixture)

      qValues <- fixture.qTable.get
      actionSpace <- fixture.env.getActionSpace
      nextQValue <- qLearning.getNextQValue(
        nextState = nextState,
        done = true,
        qValues = qValues,
        actionSpace = actionSpace
      )
    } yield nextQValue

    val result = testIO.unsafeRunSync()
    assert(result == 0.0)
  }

  test("getNextQValue returns 0.0 for unknown state-action pairs") {
    val testIO = for {
      fixture <- AgentTestFixture()
      qLearning = createQLearning(fixture)

      qValues <- fixture.qTable.get
      actionSpace <- fixture.env.getActionSpace
      nextQValue <- qLearning.getNextQValue(
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
