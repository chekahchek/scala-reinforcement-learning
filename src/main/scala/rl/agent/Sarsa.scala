package rl.agent

import cats.effect.{IO, Ref}
import rl.env.Env
import rl.logging.BaseLogger

class Sarsa[E <: Env[IO]](
    env: E,
    qTable: Ref[IO, Map[(E#State, E#Action), Double]],
    learningRate: Double,
    discountFactor: Double,
    explorationActor: IO[Exploration[E, IO]],
    logger: BaseLogger[IO]
) extends TemporalDifferenceLearning[E](
      env,
      qTable,
      learningRate,
      discountFactor,
      explorationActor,
      logger
    ) {

  override protected def getNextQValue(
      nextState: E#State,
      done: Boolean,
      qValues: Map[(E#State, E#Action), Double],
      actionSpace: List[E#Action]
  ): IO[Double] = for {
    nextAction <- act(nextState)
  } yield qValues.getOrElse((nextState, nextAction), 0.0)

}

object Sarsa {
  def apply[E <: Env[IO]](
      env: E,
      learningRate: Double = 0.1,
      discountFactor: Double = 0.9,
      exploration: IO[Exploration[E, IO]],
      logger: BaseLogger[IO]
  ): IO[Sarsa[E]] = for {
    qTable <- Ref.of[IO, Map[(E#State, E#Action), Double]](Map.empty)
    explorationActor <- exploration.map {
      case ucb @ UCB(_, _)       => IO.pure(ucb)
      case eg @ EpsilonGreedy(_) => IO.pure(eg)
    }
  } yield new Sarsa[E](
    env,
    qTable,
    learningRate,
    discountFactor,
    explorationActor,
    logger
  )
}
