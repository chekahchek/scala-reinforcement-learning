package rl.agent

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import org.scalatest.funsuite.AnyFunSuite
import scala.collection.mutable.Queue
import rl.env.GridWorld1D
import rl.logging.NoOpLogger
import rl.logging.BaseLogger
import cats.effect.Ref
import cats.effect.std.Random
import rl.agent.Exploration

class TestDoubleQLearning extends AnyFunSuite {

// Fixture for testing DoubleQLearning agents
  case class DoubleQLearningTestFixture(
      env: GridWorld1D,
      qTable1: Ref[IO, Map[(Int, Int), Double]],
      qTable2: Ref[IO, Map[(Int, Int), Double]],
      buffer: Ref[IO, Queue[(Int, Int, Double)]],
      explorationActor: IO[Exploration[GridWorld1D, IO]],
      learningRate: Double,
      discountFactor: Double,
      logger: BaseLogger[IO],
      random: Random[IO]
  )

  object DoubleQLearningTestFixture {
    def apply(
        initialQValues1: Map[(Int, Int), Double] = Map.empty,
        initialQValues2: Map[(Int, Int), Double] = Map.empty,
        explorationRate: Double = 0.0,
        learningRate: Double = 0.1,
        discountFactor: Double = 0.99
    ): IO[DoubleQLearningTestFixture] = for {
      env <- GridWorld1D(NoOpLogger)
      qTable1 <- Ref.of[IO, Map[(Int, Int), Double]](initialQValues1)
      qTable2 <- Ref.of[IO, Map[(Int, Int), Double]](initialQValues2)
      buffer <- Ref.of[IO, Queue[(Int, Int, Double)]](Queue.empty)
      explorationActor = IO.pure(new EpsilonGreedy[GridWorld1D](explorationRate))
      random <- Random.scalaUtilRandom[IO]
    } yield DoubleQLearningTestFixture(
      env,
      qTable1,
      qTable2,
      buffer,
      explorationActor,
      learningRate,
      discountFactor,
      NoOpLogger,
      random
    )
  }

  def createAgent(
      fixture: DoubleQLearningTestFixture,
      nSteps: Int = 1
  ): DoubleQLearning[GridWorld1D] = new DoubleQLearning[GridWorld1D](
    env = fixture.env,
    qTable1 = fixture.qTable1,
    qTable2 = fixture.qTable2,
    buffer = fixture.buffer,
    nSteps = nSteps,
    learningRate = fixture.learningRate,
    discountFactor = fixture.discountFactor,
    explorationActor = fixture.explorationActor,
    logger = fixture.logger,
    random = fixture.random
  )


  test("getNextQValue uses selectQValues to find best action and evalQValues to evaluate it") {
    val nextState = 3
    val testIO = for {
      fixture <- DoubleQLearningTestFixture(
        // selectQValues: action 1 has higher Q-value (1.0 > 0.5)
        initialQValues1 = Map((nextState, -1) -> 0.5, (nextState, 1) -> 1.0),
        // evalQValues: evaluate action 1 using this table (0.3)
        initialQValues2 = Map((nextState, -1) -> 0.8, (nextState, 1) -> 0.3)
      )
      agent = createAgent(fixture)

      selectQValues <- fixture.qTable1.get
      evalQValues <- fixture.qTable2.get
      actionSpace <- fixture.env.getActionSpace
      nextQValue <- agent.getNextQValue(
        nextState,
        done = false,
        selectQValues,
        evalQValues,
        actionSpace
      )
    } yield nextQValue

    val result = testIO.unsafeRunSync()
    // Best action from selectQValues is 1, eval using evalQValues gives 0.3
    assert(result == 0.3)
  }


  test("runStep transitions to next state and returns correct reward") {
    val startState = 2
    val testIO = for {
      fixture <- DoubleQLearningTestFixture()
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
      fixture <- DoubleQLearningTestFixture()
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
      fixture <- DoubleQLearningTestFixture()
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
      fixture <- DoubleQLearningTestFixture(learningRate = 0.1, discountFactor = 0.99)
      agent = createAgent(fixture, nSteps = 1)
      _ <- fixture.env.stateRef.set(state)

      // Execute a step that reaches the goal
      _ <- agent.runStep(state, action)

      // Check Q-tables were updated (at least one should have a value)
      qValues1 <- fixture.qTable1.get
      qValues2 <- fixture.qTable2.get
    } yield (qValues1.get((state, action)), qValues2.get((state, action)))

    val (qValue1, qValue2) = testIO.unsafeRunSync()
    // At least one Q-table should be updated
    assert(qValue1.isDefined || qValue2.isDefined)
    // The updated value should be positive since it leads to the goal
    val updatedValue = qValue1.orElse(qValue2).get
    assert(updatedValue > 0.0)
  }

  test("updateQValue clears buffer after update") {
    val state = 3
    val action = 1
    val testIO = for {
      fixture <- DoubleQLearningTestFixture()
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
      fixture <- DoubleQLearningTestFixture()
      agent = createAgent(fixture, nSteps = 3) // Wait for 3 steps
      _ <- fixture.env.stateRef.set(state)

      // First step (state 2 -> 1) - should not update yet
      _ <- agent.runStep(state, action)
      qValues1After1 <- fixture.qTable1.get
      qValues2After1 <- fixture.qTable2.get
      bufferSize1 <- fixture.buffer.get.map(_.size)

      // Second step (state 1 -> 0) - still should not update
      nextState1 <- fixture.env.getState
      _ <- agent.runStep(nextState1, action)
      qValues1After2 <- fixture.qTable1.get
      qValues2After2 <- fixture.qTable2.get
      bufferSize2 <- fixture.buffer.get.map(_.size)
    } yield (
      qValues1After1.isEmpty && qValues2After1.isEmpty,
      bufferSize1,
      qValues1After2.isEmpty && qValues2After2.isEmpty,
      bufferSize2
    )

    val (emptyAfter1, size1, emptyAfter2, size2) = testIO.unsafeRunSync()
    assert(emptyAfter1) // No Q-value update after 1 step
    assert(size1 == 1)
    assert(emptyAfter2) // No Q-value update after 2 steps
    assert(size2 == 2)
  }

  test("updateQValue computes n-step return correctly") {
    val state = 2
    val testIO = for {
      fixture <- DoubleQLearningTestFixture(
        learningRate = 1.0, // Set to 1.0 for easier verification
        discountFactor = 0.9
      )
      agent = createAgent(fixture, nSteps = 2)
      _ <- fixture.env.stateRef.set(state)

      // First step (state=2 -> state=3, reward=0)
      _ <- agent.runStep(state, 1)
      // Second step triggers update (state=3 -> state=4, reward=1)
      _ <- agent.runStep(3, 1)

      qValues1 <- fixture.qTable1.get
      qValues2 <- fixture.qTable2.get
    } yield (qValues1.get((state, 1)), qValues2.get((state, 1)))

    val (qValue1, qValue2) = testIO.unsafeRunSync()
    // At least one Q-table should be updated
    assert(qValue1.isDefined || qValue2.isDefined)
    val qValue = qValue1.orElse(qValue2).get
    // n-step return = 0 + 0.9 * 1.0 = 0.9 (first reward is 0, second reward is 1.0)
    assert(Math.abs(qValue - 0.9) < 0.01)
  }

  // ==================== runEpisode tests ====================

  test("runEpisode completes and returns positive reward when reaching goal") {
    val testIO = for {
      fixture <- DoubleQLearningTestFixture(
        learningRate = 0.5,
        discountFactor = 0.99,
        explorationRate = 0.0
      )
      // Pre-populate Q-tables with values that guide agent to goal
      _ <- fixture.qTable1.set(Map(
        (2, 1) -> 1.0, (2, -1) -> 0.0,
        (3, 1) -> 1.0, (3, -1) -> 0.0
      ))
      _ <- fixture.qTable2.set(Map(
        (2, 1) -> 1.0, (2, -1) -> 0.0,
        (3, 1) -> 1.0, (3, -1) -> 0.0
      ))
      agent = createAgent(fixture)
      result <- agent.runEpisode()
    } yield result

    val (totalReward, stepCount) = testIO.unsafeRunSync()
    assert(totalReward > 0.0) // Should get positive reward for reaching goal
    assert(stepCount > 0) // Should take at least one step
  }


  // ==================== Integration tests ====================

  test("agent can learn to reach goal through multiple episodes") {
    val testIO = for {
      fixture <- DoubleQLearningTestFixture(
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

      qValues1 <- fixture.qTable1.get
      qValues2 <- fixture.qTable2.get
    } yield (result1._1, result2._1, result2._2, qValues1, qValues2)

    val (done1, done2, finalState, qValues1, qValues2) = testIO.unsafeRunSync()
    assert(!done1)
    assert(done2)
    assert(finalState == 4)
    // At least one Q-table should have positive Q-value for (3, 1) since it leads to goal
    val qValueFor3_1 = qValues1.getOrElse((3, 1), 0.0) + qValues2.getOrElse((3, 1), 0.0)
    assert(qValueFor3_1 > 0.0)
  }

  test("agent handles boundary conditions at grid edges") {
    val testIO = for {
      fixture <- DoubleQLearningTestFixture()
      agent = createAgent(fixture)
      _ <- fixture.env.stateRef.set(0) // At left edge
      result <- agent.runStep(0, -1) // Try to move left (should stay at 0)
    } yield result._2

    val nextState = testIO.unsafeRunSync()
    assert(nextState == 0) // Should stay at 0 (boundary)
  }

  test("double Q-learning uses different tables for action selection and evaluation") {
    // This test verifies the core double Q-learning behavior:
    // One table selects the best action, the other evaluates it
    val state = 2
    val testIO = for {
      fixture <- DoubleQLearningTestFixture(
        // Table 1 prefers action 1
        initialQValues1 = Map((state, -1) -> 0.0, (state, 1) -> 1.0),
        // Table 2 has different values
        initialQValues2 = Map((state, -1) -> 0.5, (state, 1) -> 0.2)
      )
      agent = createAgent(fixture)

      selectQValues <- fixture.qTable1.get
      evalQValues <- fixture.qTable2.get
      actionSpace <- fixture.env.getActionSpace

      // When using table1 to select and table2 to evaluate:
      // Best action from table1 is 1 (Q=1.0)
      // Evaluation from table2 for action 1 is 0.2
      nextQ <- agent.getNextQValue(state, done = false, selectQValues, evalQValues, actionSpace)
    } yield nextQ

    val result = testIO.unsafeRunSync()
    assert(result == 0.2) // Should be evaluated using table2's value for the selected action
  }
}
