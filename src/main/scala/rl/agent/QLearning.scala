package rl.agent

import cats.effect.{IO, Ref}
import rl.env.Env
import rl.logging.BaseLogger

class QLearning[E <: Env[IO]](val env: E,
                              qTable: Ref[IO, Map[(E#State, E#Action), Double]],
                              learningRate: Double,
                              discountFactor: Double,
                              explorationActor: IO[Exploration[E, IO]],
                              logger: BaseLogger[IO]
                             ) {


  def act(state: E#State): IO[E#Action] = for {
    actionSpace <- env.getActionSpace
    qValues <- qTable.get
    action <- explorationActor.flatMap { a => a.getAction(actionSpace, state, qValues) }
  } yield action

  def runStep(state: E#State): IO[Boolean] = for {
    action <- act(state)
    res <- env.step(action.asInstanceOf[env.Action])
    nextState = res._1
    reward = res._2
    done = res._3
    nextStateStr <- env.renderState(nextState)
    _ <- logger.debug(s"Agent took action: $action resulting in state: $nextStateStr, reward: ${res._2}, done: ${res._3}")

    qValues <- qTable.get
    actionSpace <- env.getActionSpace

    // Update Q-values
    currentQ = qValues.getOrElse((state, action), 0.0)
    maxNextQ = if(done) 0.0 else actionSpace.map(a => qValues.getOrElse((nextState, a), 0.0)).max
    updatedQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ)
    _ <- qTable.update(qv => qv + ((state, action) -> updatedQ))
  } yield done

  def runEpisode(): IO[Unit] = {
    def loop(state: E#State): IO[Unit] = for {
      done <- runStep(state)
      _ <- if(done) IO.unit
      else for {
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


  def learn(numEpisodes: Int): IO[Unit] = {
    def loop(episode: Int): IO[Unit] = {
      if(episode >= numEpisodes) IO.unit
      else for {
        _ <- runEpisode()
        _ <- logger.debug(s"Completed episode: ${episode + 1}")
        _ <- loop(episode + 1)
      } yield ()
    }

    loop(0)
  }


}


object QLearning {
  def apply[E <: Env[IO]](env: E,
                          learningRate: Double = 0.1,
                          discountFactor: Double = 0.9,
                          exploration: IO[Exploration[E, IO]],
                          logger: BaseLogger[IO]
                         ): IO[QLearning[E]] = for {
    qTable <- Ref.of[IO, Map[(E#State, E#Action), Double]](Map.empty)
    explorationActor <- exploration.map {
      case ucb@UCB(_, _) => IO.pure(ucb)
      case eg@EpsilonGreedy(_) => IO.pure(eg)
    }
  } yield new QLearning[E](env, qTable, learningRate, discountFactor, explorationActor, logger)
}
