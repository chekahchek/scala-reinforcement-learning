package rl.agent

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import org.scalatest.funsuite.AnyFunSuite
import scala.collection.mutable.Queue
import rl.env.GridWorld1D
import rl.logging.NoOpLogger
import rl.TestFixture.AgentTestFixture

class TestTemporalDifferenceLearning extends AnyFunSuite {

  def createAgent(
      fixture: AgentTestFixture,
      nSteps: Int = 1
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


  test("act selects the action with highest Q-value when exploiting") {
    val state = 2
    val testIO = for {
      fixture <- AgentTestFixture(
        initialQValues = Map((state, -1) -> 0.5, (state, 1) -> 2.0),
        explorationRate = 0.0
      )
      agent = createAgent(fixture)
      _ <- fixture.env.stateRef.set(state)
      action <- agent.act(state)
    } yield action

    val result = testIO.unsafeRunSync()
    assert(result == 1)
  }


  test("peekAct does not increment UCB count when called multiple times") {
    val state = 2
    val testIO = for {
      fixture <- AgentTestFixture(
        initialQValues = Map((state, -1) -> 0.0, (state, 1) -> 1.0),
        explorationRate = 0.0
      )
      agent = createAgent(fixture)
      peekAction1 <- agent.peekAct(state)
      peekAction2 <- agent.peekAct(state)
    } yield (peekAction1, peekAction2)

    val result = testIO.unsafeRunSync()
    assert(result._1 == result._2)
  }

  test("runStep transitions to next state and returns correct reward") {
    val startState = 2
    val testIO = for {
      fixture <- AgentTestFixture()
      agent = createAgent(fixture)
      _ <- fixture.env.stateRef.set(startState)
      result <- agent.runStep(startState, 1)
    } yield result

    val (done, nextState, reward) = testIO.unsafeRunSync()
    assert(nextState == 3)
    assert(reward == 0.0) // Not at target yet
    assert(!done)
  }

  test("runStep returns done=true and reward=1.0 when reaching target") {
    val startState = 3 // One step away from target (4)
    val testIO = for {
      fixture <- AgentTestFixture()
      agent = createAgent(fixture)
      _ <- fixture.env.stateRef.set(startState)
      result <- agent.runStep(startState, 1) // Move right to target
    } yield result

    val (done, nextState, reward) = testIO.unsafeRunSync()
    assert(nextState == 4)
    assert(reward == 1.0)
    assert(done)
  }

  test("runStep adds transition to buffer") {
    val state = 2
    val action = -1 // Move left to avoid reaching goal
    val testIO = for {
      fixture <- AgentTestFixture()
      agent = createAgent(fixture, nSteps = 3) // Use nSteps > 1 so buffer doesn't clear immediately
      _ <- fixture.env.stateRef.set(state)
      sizeBefore <- fixture.buffer.get.map(_.size)
      _ <- agent.runStep(state, action)
      sizeAfter <- fixture.buffer.get.map(_.size)
    } yield (sizeBefore, sizeAfter)

    val (sizeBefore, sizeAfter) = testIO.unsafeRunSync()
    assert(sizeBefore == 0, s"Expected buffer to start empty, but had size $sizeBefore")
    assert(sizeAfter == 1, s"Expected buffer size 1 after runStep, but had size $sizeAfter")
  }


  test("updateQValue updates Q-table when episode is done") {
    val state = 3
    val action = 1
    val testIO = for {
      fixture <- AgentTestFixture(learningRate = 0.1, discountFactor = 0.99)
      agent = createAgent(fixture, nSteps = 1)
      _ <- fixture.env.stateRef.set(state)

      // Execute a step that reaches the goal
      _ <- agent.runStep(state, action)

      // Check Q-table was updated
      qValues <- fixture.qTable.get
    } yield qValues.get((state, action))

    val qValue = testIO.unsafeRunSync()
    assert(qValue.isDefined)
    assert(qValue.get > 0.0) // Q-value should be updated with positive reward
  }

  test("updateQValue clears buffer after update") {
    val state = 3
    val action = 1
    val testIO = for {
      fixture <- AgentTestFixture()
      agent = createAgent(fixture, nSteps = 1)
      _ <- fixture.env.stateRef.set(state)

      // Execute a step that triggers an update (reaches goal)
      _ <- agent.runStep(state, action)

      bufferAfter <- fixture.buffer.get
    } yield bufferAfter.size

    val bufferSize = testIO.unsafeRunSync()
    assert(bufferSize == 0) // Buffer should be cleared after update
  }

  test("updateQValue waits for n steps before updating") {
    val state = 2
    val action = -1 // Move left to avoid reaching goal (state 4)
    val testIO = for {
      fixture <- AgentTestFixture()
      agent = createAgent(fixture, nSteps = 3) // Wait for 3 steps
      _ <- fixture.env.stateRef.set(state)

      // First step (state 2 -> 1) - should not update yet
      _ <- agent.runStep(state, action)
      qValuesAfter1 <- fixture.qTable.get
      bufferSize1 <- fixture.buffer.get.map(_.size)

      // Second step (state 1 -> 0) - still should not update
      nextState1 <- fixture.env.getState
      _ <- agent.runStep(nextState1, action)
      qValuesAfter2 <- fixture.qTable.get
      bufferSize2 <- fixture.buffer.get.map(_.size)
    } yield (qValuesAfter1.isEmpty, bufferSize1, qValuesAfter2.isEmpty, bufferSize2)

    val (emptyAfter1, size1, emptyAfter2, size2) = testIO.unsafeRunSync()
    assert(emptyAfter1) // No Q-value update after 1 step
    assert(size1 == 1)
    assert(emptyAfter2) // No Q-value update after 2 steps
    assert(size2 == 2)
  }

  test("updateQValue computes n-step return correctly") {
    val state = 2
    val testIO = for {
      fixture <- AgentTestFixture(
        learningRate = 1.0, // Set to 1.0 for easier verification
        discountFactor = 0.9
      )
      agent = createAgent(fixture, nSteps = 2)
      _ <- fixture.env.stateRef.set(state)

      // First step (state=2 -> state=3, reward=0)
      _ <- agent.runStep(state, 1)
      // Second step triggers update (state=3 -> state=4, reward=1)
      _ <- agent.runStep(3, 1)

      qValues <- fixture.qTable.get
    } yield qValues.get((state, 1))

    val qValue = testIO.unsafeRunSync()
    assert(qValue.isDefined)
    // n-step return = 0 + 0.9 * 1.0 = 0.9 (first reward is 0, second reward is 1.0)
    assert(Math.abs(qValue.get - 0.9) < 0.01)
  }

  // ==================== Integration tests ====================

  test("agent can learn to reach goal through multiple episodes") {
    val testIO = for {
      fixture <- AgentTestFixture(
        learningRate = 0.5,
        discountFactor = 0.99,
        explorationRate = 0.0
      )
      agent = createAgent(fixture, nSteps = 1)

      // Run a simple episode: start at 2, move right to reach 4
      _ <- fixture.env.reset()
      state0 <- fixture.env.getState // state = 2
      result1 <- agent.runStep(state0, 1) // state = 3
      result2 <- agent.runStep(result1._2, 1) // state = 4 (goal)

      qValues <- fixture.qTable.get
    } yield (result1._1, result2._1, result2._2, qValues)

    val (done1, done2, finalState, qValues) = testIO.unsafeRunSync()
    assert(!done1)
    assert(done2)
    assert(finalState == 4)
    // Q-value for (3, 1) should be positive since it leads to goal
    assert(qValues.getOrElse((3, 1), 0.0) > 0.0)
  }

  test("agent handles boundary conditions at grid edges") {
    val testIO = for {
      fixture <- AgentTestFixture()
      agent = createAgent(fixture)
      _ <- fixture.env.stateRef.set(0) // At left edge
      result <- agent.runStep(0, -1) // Try to move left (should stay at 0)
    } yield result._2

    val nextState = testIO.unsafeRunSync()
    assert(nextState == 0) // Should stay at 0 (boundary)
  }
}
