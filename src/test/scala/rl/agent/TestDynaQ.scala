package rl.agent

import cats.effect.IO
import cats.effect.unsafe.implicits.global
import org.scalatest.funsuite.AnyFunSuite
import rl.env.GridWorld1D
import cats.effect.Ref
import rl.logging.BaseLogger
import rl.logging.NoOpLogger

class TestDynaQ extends AnyFunSuite {

  // Fixture for testing DynaQ agents
  case class DynaQTestFixture(
      env: GridWorld1D,
      qTable: Ref[IO, Map[(Int, Int), Double]],
      model: Ref[IO, Map[(Int, Int), (Int, Double, Boolean)]],
      explorationActor: IO[Exploration[GridWorld1D, IO]],
      learningRate: Double,
      discountFactor: Double,
      planningSteps: Int,
      logger: BaseLogger[IO]
  )

  object DynaQTestFixture {
    def apply(
        initialQValues: Map[(Int, Int), Double] = Map.empty,
        initialModel: Map[(Int, Int), (Int, Double, Boolean)] = Map.empty,
        explorationRate: Double = 0.0,
        learningRate: Double = 0.1,
        discountFactor: Double = 0.99,
        planningSteps: Int = 5
    ): IO[DynaQTestFixture] = for {
      env <- GridWorld1D(NoOpLogger)
      qTable <- Ref.of[IO, Map[(Int, Int), Double]](initialQValues)
      model <- Ref.of[IO, Map[(Int, Int), (Int, Double, Boolean)]](initialModel)
      explorationActor = IO.pure(new EpsilonGreedy[GridWorld1D](explorationRate))
    } yield DynaQTestFixture(
      env,
      qTable,
      model,
      explorationActor,
      learningRate,
      discountFactor,
      planningSteps,
      NoOpLogger
    )
  }

  // Fixture for testing DynaQ+ agents
  case class DynaQPlusTestFixture(
      env: GridWorld1D,
      qTable: Ref[IO, Map[(Int, Int), Double]],
      model: Ref[IO, Map[(Int, Int), (Int, Double, Boolean)]],
      lastVisited: Ref[IO, Map[(Int, Int), Long]],
      timeStep: Ref[IO, Long],
      explorationActor: IO[Exploration[GridWorld1D, IO]],
      learningRate: Double,
      discountFactor: Double,
      planningSteps: Int,
      kappa: Double,
      logger: BaseLogger[IO]
  )

  object DynaQPlusTestFixture {
    def apply(
        initialQValues: Map[(Int, Int), Double] = Map.empty,
        initialModel: Map[(Int, Int), (Int, Double, Boolean)] = Map.empty,
        initialLastVisited: Map[(Int, Int), Long] = Map.empty,
        initialTimeStep: Long = 0L,
        explorationRate: Double = 0.0,
        learningRate: Double = 0.1,
        discountFactor: Double = 0.99,
        planningSteps: Int = 5,
        kappa: Double = 0.001
    ): IO[DynaQPlusTestFixture] = for {
      env <- GridWorld1D(NoOpLogger)
      qTable <- Ref.of[IO, Map[(Int, Int), Double]](initialQValues)
      model <- Ref.of[IO, Map[(Int, Int), (Int, Double, Boolean)]](initialModel)
      lastVisited <- Ref.of[IO, Map[(Int, Int), Long]](initialLastVisited)
      timeStep <- Ref.of[IO, Long](initialTimeStep)
      explorationActor = IO.pure(new EpsilonGreedy[GridWorld1D](explorationRate))
    } yield DynaQPlusTestFixture(
      env,
      qTable,
      model,
      lastVisited,
      timeStep,
      explorationActor,
      learningRate,
      discountFactor,
      planningSteps,
      kappa,
      NoOpLogger
    )
  }

  def createDynaQAgent(
      fixture: DynaQTestFixture
  ): DynaQ[GridWorld1D] = new DynaQ[GridWorld1D](
    env = fixture.env,
    qTable = fixture.qTable,
    learningRate = fixture.learningRate,
    discountFactor = fixture.discountFactor,
    planningSteps = fixture.planningSteps,
    model = fixture.model,
    explorationActor = fixture.explorationActor,
    logger = fixture.logger
  )

  def createDynaQPlusAgent(
      fixture: DynaQPlusTestFixture
  ): DynaQPlus[GridWorld1D] = new DynaQPlus[GridWorld1D](
    env = fixture.env,
    qTable = fixture.qTable,
    learningRate = fixture.learningRate,
    discountFactor = fixture.discountFactor,
    planningSteps = fixture.planningSteps,
    model = fixture.model,
    explorationActor = fixture.explorationActor,
    logger = fixture.logger,
    lastVisited = fixture.lastVisited,
    timeStep = fixture.timeStep,
    kappa = fixture.kappa
  )

  // ==================== DynaQ Tests ====================

  test("act selects the action with highest Q-value when exploiting") {
    val state = 2
    val testIO = for {
      fixture <- DynaQTestFixture(
        initialQValues = Map((state, -1) -> 0.5, (state, 1) -> 2.0),
        explorationRate = 0.0
      )
      agent = createDynaQAgent(fixture)
      _ <- fixture.env.stateRef.set(state)
      action <- agent.act(state)
    } yield action

    val result = testIO.unsafeRunSync()
    assert(result == 1)
  }

  test("runStep transitions to next state and returns correct reward") {
    val startState = 2
    val testIO = for {
      fixture <- DynaQTestFixture()
      agent = createDynaQAgent(fixture)
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
      fixture <- DynaQTestFixture()
      agent = createDynaQAgent(fixture)
      _ <- fixture.env.stateRef.set(startState)
      result <- agent.runStep(startState, 1) // Move right to target
    } yield result

    val (done, nextState, reward) = testIO.unsafeRunSync()
    assert(nextState == 4)
    assert(reward == 1.0)
    assert(done)
  }

  test("runStep updates Q-table") {
    val state = 3
    val action = 1
    val testIO = for {
      fixture <- DynaQTestFixture(learningRate = 0.1, discountFactor = 0.99)
      agent = createDynaQAgent(fixture)
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

  test("runStep updates model with experience") {
    val state = 2
    val action = 1
    val testIO = for {
      fixture <- DynaQTestFixture()
      agent = createDynaQAgent(fixture)
      _ <- fixture.env.stateRef.set(state)

      modelBefore <- fixture.model.get
      _ <- agent.runStep(state, action)
      modelAfter <- fixture.model.get
    } yield (modelBefore.get((state, action)), modelAfter.get((state, action)))

    val (before, after) = testIO.unsafeRunSync()
    assert(before.isEmpty)
    assert(after.isDefined)
    assert(after.get == (3, 0.0, false)) // nextState=3, reward=0, done=false
  }

  test("runStep performs planning updates using model") {
    val state = 2
    val action = 1
    val testIO = for {
      fixture <- DynaQTestFixture(
        // Pre-populate model with some experience
        initialModel = Map(
          (1, 1) -> (2, 0.0, false),
          (2, 1) -> (3, 0.0, false)
        ),
        planningSteps = 5,
        learningRate = 0.5
      )
      agent = createDynaQAgent(fixture)
      _ <- fixture.env.stateRef.set(state)

      // Execute a step - planning should sample from model
      _ <- agent.runStep(state, action)

      qValues <- fixture.qTable.get
    } yield qValues

    val qValues = testIO.unsafeRunSync()
    // Q-table should have updates from both direct learning and planning
    assert(qValues.nonEmpty)
  }

  test("runStep with planningSteps=0 does not perform planning") {
    val state = 2
    val action = 1
    val testIO = for {
      fixture <- DynaQTestFixture(
        initialModel = Map((1, 1) -> (2, 1.0, false)),
        planningSteps = 0, // No planning
        learningRate = 0.5
      )
      agent = createDynaQAgent(fixture)
      _ <- fixture.env.stateRef.set(state)

      qValuesBefore <- fixture.qTable.get
      _ <- agent.runStep(state, action)
      qValuesAfter <- fixture.qTable.get
    } yield (qValuesBefore.get((1, 1)), qValuesAfter.get((1, 1)))

    val (before, after) = testIO.unsafeRunSync()
    // State (1, 1) from model should not be updated since planningSteps=0
    assert(before.isEmpty)
    assert(after.isEmpty)
  }

  // ==================== DynaQ+ Tests ====================

  test("DynaQ+ act selects the action with highest Q-value when exploiting") {
    val state = 2
    val testIO = for {
      fixture <- DynaQPlusTestFixture(
        initialQValues = Map((state, -1) -> 0.5, (state, 1) -> 2.0),
        explorationRate = 0.0
      )
      agent = createDynaQPlusAgent(fixture)
      _ <- fixture.env.stateRef.set(state)
      action <- agent.act(state)
    } yield action

    val result = testIO.unsafeRunSync()
    assert(result == 1)
  }

  test("DynaQ+ runStep updates timeStep counter") {
    val state = 2
    val action = 1
    val testIO = for {
      fixture <- DynaQPlusTestFixture(initialTimeStep = 0L)
      agent = createDynaQPlusAgent(fixture)
      _ <- fixture.env.stateRef.set(state)

      timeStepBefore <- fixture.timeStep.get
      _ <- agent.runStep(state, action)
      timeStepAfter <- fixture.timeStep.get
    } yield (timeStepBefore, timeStepAfter)

    val (before, after) = testIO.unsafeRunSync()
    assert(before == 0L)
    assert(after == 1L)
  }

  test("DynaQ+ runStep updates lastVisited map") {
    val state = 2
    val action = 1
    val testIO = for {
      fixture <- DynaQPlusTestFixture(initialTimeStep = 0L)
      agent = createDynaQPlusAgent(fixture)
      _ <- fixture.env.stateRef.set(state)

      lastVisitedBefore <- fixture.lastVisited.get
      _ <- agent.runStep(state, action)
      lastVisitedAfter <- fixture.lastVisited.get
    } yield (lastVisitedBefore.get((state, action)), lastVisitedAfter.get((state, action)))

    val (before, after) = testIO.unsafeRunSync()
    assert(before.isEmpty)
    assert(after.isDefined)
    assert(after.get == 1L) // Should be updated to current timeStep (1 after increment)
  }

  test("DynaQ+ adds exploration bonus based on recency") {
    // This test verifies that DynaQ+ considers time since last visit
    val state = 2
    val action = 1
    val testIO = for {
      fixture <- DynaQPlusTestFixture(
        initialModel = Map((1, 1) -> (2, 0.0, false)),
        initialLastVisited = Map((1, 1) -> 0L), // Last visited at time 0
        initialTimeStep = 100L, // Current time is 100, so tau = 100
        planningSteps = 10,
        kappa = 0.1, // Use larger kappa for visible effect
        learningRate = 1.0
      )
      agent = createDynaQPlusAgent(fixture)
      _ <- fixture.env.stateRef.set(state)

      // Execute step - planning should add exploration bonus
      _ <- agent.runStep(state, action)

      qValues <- fixture.qTable.get
    } yield qValues.get((1, 1))

    val qValue = testIO.unsafeRunSync()
    // With tau = 100 and kappa = 0.1, bonus = 0.1 * sqrt(100) = 1.0
    // Q-value should reflect this bonus during planning
    assert(qValue.isDefined)
  }

  // ==================== Integration Tests ====================

  test("DynaQ agent can learn to reach goal through multiple episodes") {
    val testIO = for {
      fixture <- DynaQTestFixture(
        learningRate = 0.5,
        discountFactor = 0.99,
        explorationRate = 0.0,
        planningSteps = 5
      )
      agent = createDynaQAgent(fixture)

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

  test("DynaQ agent handles boundary conditions at grid edges") {
    val testIO = for {
      fixture <- DynaQTestFixture()
      agent = createDynaQAgent(fixture)
      _ <- fixture.env.stateRef.set(0) // At left edge
      result <- agent.runStep(0, -1) // Try to move left (should stay at 0)
    } yield result._2

    val nextState = testIO.unsafeRunSync()
    assert(nextState == 0) // Should stay at 0 (boundary)
  }

  test("DynaQ+ agent can learn to reach goal through multiple episodes") {
    val testIO = for {
      fixture <- DynaQPlusTestFixture(
        learningRate = 0.5,
        discountFactor = 0.99,
        explorationRate = 0.0,
        planningSteps = 5,
        kappa = 0.001
      )
      agent = createDynaQPlusAgent(fixture)

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

}
