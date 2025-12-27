package rl.agent

import cats.effect.{IO, Ref}
import rl.env.Env
import rl.logging.BaseLogger

abstract class TemporalDifferenceLearning[E <: Env[IO]](
    val env: E,
    protected val qTable: Ref[IO, Map[(E#State, E#Action), Double]],
    protected val learningRate: Double,
    protected val discountFactor: Double,
    protected val explorationActor: IO[Exploration[E, IO]],
    protected val logger: BaseLogger[IO]
) {

  def act(state: E#State): IO[E#Action] = for {
    actionSpace <- env.getActionSpace
    qValues <- qTable.get
    action <- explorationActor.flatMap { a =>
      a.getAction(actionSpace, state, qValues)
    }
  } yield action

// To be overridden by the subclass
  protected def getNextQValue(
      nextState: E#State,
      done: Boolean,
      qValues: Map[(E#State, E#Action), Double],
      actionSpace: List[E#Action]
  ): IO[Double]

  protected def runStep(state: E#State, action: E#Action): IO[Boolean] = for {
    res <- env.step(action.asInstanceOf[env.Action])
    nextState = res._1
    reward = res._2
    done = res._3

    qValues <- qTable.get
    actionSpace <- env.getActionSpace

    // Update Q-values
    currentQ = qValues.getOrElse((state, action), 0.0)
    nextQ <- getNextQValue(nextState, done, qValues, actionSpace)
    updatedQ =
      currentQ + learningRate * (reward + discountFactor * nextQ - currentQ)
    _ <- qTable.update(qv => qv + ((state, action) -> updatedQ))
  } yield done

  def runEpisode(): IO[Unit] = {
    def loop(state: E#State): IO[Unit] = for {
      action <- act(state)
      done <- runStep(state, action)
      _ <-
        if (done) IO.unit
        else
          for {
            nextState <- env.getState
            _ <- loop(nextState)
          } yield ()
    } yield ()

    for {
      _ <- env.reset()
      initialState <- env.getState
      _ <- loop(initialState)
    } yield ()
  }

  def learn(episodes: Int): IO[Unit] = {
    def loop(episode: Int): IO[Unit] = {
      if (episode >= episodes) IO.unit
      else
        for {
          _ <- runEpisode()
          _ <- logger.info(s"Completed episode: ${episode + 1}")
          _ <- loop(episode + 1)
        } yield ()
    }

    loop(0)
  }

}
