package rl.agent

import cats.effect.{IO, Ref}
import scala.util.Random
import rl.env.Env
import rl.logging.BaseLogger

abstract class BaseDynaQ[E <: Env[IO]](
    val env: E,
    protected val qTable: Ref[IO, Map[(E#State, E#Action), Double]],
    protected val learningRate: Double,
    protected val discountFactor: Double,
    protected val planningSteps: Int,
    // Model stores the latest experience (state, action) -> (nextState, reward, done)
    protected val model: Ref[IO, Map[(E#State, E#Action), (E#State, Double, Boolean)]],
    protected val explorationActor: IO[Exploration[E, IO]],
    protected val logger: BaseLogger[IO]
) extends Agent[E] {

  def act(state: E#State): IO[E#Action] = for {
    actionSpace <- env.getActionSpace
    qValues <- qTable.get
    action <- explorationActor.flatMap { a =>
      a.getAction(actionSpace, state, qValues)
    }
  } yield action

  protected def updateQValue(
      done: Boolean,
      prevState: E#State,
      prevAction: E#Action,
      reward: Double,
      nextState: E#State
  ): IO[Unit] = {
    for {
      qValues <- qTable.get
      actionSpace <- env.getActionSpace

      currentQ = qValues.getOrElse((prevState, prevAction), 0.0)
      nextQ =
        if (done) 0.0
        else
          actionSpace.map(a => qValues.getOrElse((nextState, a), 0.0)).max

      updatedQ = currentQ + learningRate * (reward + discountFactor * nextQ - currentQ)
      _ <- qTable.update(qv => qv + ((prevState, prevAction) -> updatedQ))
    } yield ()
  }

  // Override in subclasses for DynaQ+
  protected def computePlanningReward(
      state: E#State,
      action: E#Action,
      baseReward: Double
  ): IO[Double] = IO.pure(baseReward)

  // Override in subclasses for DynaQ+
  protected def onStepTaken(state: E#State, action: E#Action): IO[Unit] = IO.unit

  def runStep(
      state: E#State,
      action: E#Action
  ): IO[(Boolean, E#State, Double)] = {
    def planningLoop(step: Int): IO[Unit] = {
      if (step >= planningSteps) IO.unit
      else
        for {
          // Sample a random state-action pair from the model
          modelMap <- model.get
          keys = modelMap.keys.toSeq
          _ <-
            if (keys.isEmpty) IO.unit
            else {
              val (sampledState, sampledAction) = Random.shuffle(keys).head
              // Get the value (nextState, reward, done) from the model
              val (nextState, reward, wasDone) = modelMap.getOrElse(
                (sampledState, sampledAction),
                (sampledState, 0.0, false)
              )

              for {
                augmentedReward <- computePlanningReward(
                  sampledState,
                  sampledAction,
                  reward
                )
                _ <- updateQValue(
                  wasDone,
                  sampledState,
                  sampledAction,
                  augmentedReward,
                  nextState
                )
                _ <- planningLoop(step + 1)
              } yield ()
            }
        } yield ()
    }

    for {
      res <- env.step(action.asInstanceOf[env.Action])
      nextState = res._1.asInstanceOf[E#State]
      reward = res._2
      done = res._3

      // Hook for subclasses for DynaQ+ to track state
      _ <- onStepTaken(state, action)

      _ <- updateQValue(done, state, action, reward, nextState)

      // Update model
      _ <- model.update(m => m + ((state, action) -> (nextState, reward, done)))

      // Planning
      _ <- planningLoop(0)
    } yield (done, nextState, reward)
  }
}

// Concrete implementation of DynaQ
class DynaQ[E <: Env[IO]](
    env: E,
    qTable: Ref[IO, Map[(E#State, E#Action), Double]],
    learningRate: Double,
    discountFactor: Double,
    planningSteps: Int,
    model: Ref[IO, Map[(E#State, E#Action), (E#State, Double, Boolean)]],
    explorationActor: IO[Exploration[E, IO]],
    logger: BaseLogger[IO]
) extends BaseDynaQ[E](
      env,
      qTable,
      learningRate,
      discountFactor,
      planningSteps,
      model,
      explorationActor,
      logger
    )

object DynaQ {
  def apply[E <: Env[IO]](
      env: E,
      learningRate: Double,
      discountFactor: Double,
      planningSteps: Int,
      exploration: IO[Exploration[E, IO]],
      logger: BaseLogger[IO]
  ): IO[DynaQ[E]] = for {
    qTable <- Ref.of[IO, Map[(E#State, E#Action), Double]](Map.empty)
    model <- Ref.of[IO, Map[(E#State, E#Action), (E#State, Double, Boolean)]](
      Map.empty
    )
    explorationActor <- exploration.map {
      case ucb @ UCB(_, _)       => IO.pure(ucb)
      case eg @ EpsilonGreedy(_) => IO.pure(eg)
    }
  } yield new DynaQ[E](
    env,
    qTable,
    learningRate,
    discountFactor,
    planningSteps,
    model,
    explorationActor,
    logger
  )
}

/** Concrete implementation of Dyna-Q+ agent with exploration bonuses based on recency of visits.
  *
  * During planning, an exploration bonus κ√τ is added to the reward, where τ is
  * the number of time steps since the state-action pair was last visited, and κ
  * is a small constant (typically 0.001).
  *
  * This encourages the agent to revisit state-action pairs that haven't been
  * tried recently, which is particularly useful in non-stationary environments.
  */
class DynaQPlus[E <: Env[IO]](
    env: E,
    qTable: Ref[IO, Map[(E#State, E#Action), Double]],
    learningRate: Double,
    discountFactor: Double,
    planningSteps: Int,
    model: Ref[IO, Map[(E#State, E#Action), (E#State, Double, Boolean)]],
    explorationActor: IO[Exploration[E, IO]],
    logger: BaseLogger[IO],
    private val lastVisited: Ref[IO, Map[(E#State, E#Action), Long]],
    private val timeStep: Ref[IO, Long],
    private val kappa: Double
) extends BaseDynaQ[E](
      env,
      qTable,
      learningRate,
      discountFactor,
      planningSteps,
      model,
      explorationActor,
      logger
    ) {

  override protected def computePlanningReward(
      state: E#State,
      action: E#Action,
      baseReward: Double
  ): IO[Double] = for {
    currentTime <- timeStep.get
    lastVisitedMap <- lastVisited.get
    tau = currentTime - lastVisitedMap.getOrElse((state, action), 0L)
    bonus = kappa * math.sqrt(tau.toDouble)
  } yield baseReward + bonus

  override protected def onStepTaken(
      state: E#State,
      action: E#Action
  ): IO[Unit] = for {
    currentTime <- timeStep.updateAndGet(_ + 1)
    _ <- lastVisited.update(m => m + ((state, action) -> currentTime))
  } yield ()
}

object DynaQPlus {

  def apply[E <: Env[IO]](
      env: E,
      learningRate: Double,
      discountFactor: Double,
      planningSteps: Int,
      exploration: IO[Exploration[E, IO]],
      logger: BaseLogger[IO],
      kappa: Double
  ): IO[DynaQPlus[E]] = for {
    qTable <- Ref.of[IO, Map[(E#State, E#Action), Double]](Map.empty)
    model <- Ref.of[IO, Map[(E#State, E#Action), (E#State, Double, Boolean)]](
      Map.empty
    )
    lastVisited <- Ref.of[IO, Map[(E#State, E#Action), Long]](Map.empty)
    timeStep <- Ref.of[IO, Long](0L)
    explorationActor <- exploration.map {
      case ucb @ UCB(_, _)       => IO.pure(ucb)
      case eg @ EpsilonGreedy(_) => IO.pure(eg)
    }
  } yield new DynaQPlus[E](
    env,
    qTable,
    learningRate,
    discountFactor,
    planningSteps,
    model,
    explorationActor,
    logger,
    lastVisited,
    timeStep,
    kappa
  )
}
