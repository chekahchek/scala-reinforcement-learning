package rl.agent

import cats.effect.{IO, Ref}
import scala.collection.mutable.Queue
import rl.env.Env
import rl.logging.BaseLogger

class QLearning[E <: Env[IO]](
    env: E,
    qTable: Ref[IO, Map[(E#State, E#Action), Double]],
    buffer: Ref[IO, Queue[(E#State, E#Action, Double)]],
    nSteps: Int,
    learningRate: Double,
    discountFactor: Double,
    explorationActor: IO[Exploration[E, IO]],
    logger: BaseLogger[IO]
) extends TemporalDifferenceLearning[E](
      env,
      qTable,
      buffer,
      nSteps,
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
  ): IO[Double] = IO.pure {
    if (done) 0.0
    else actionSpace.map(a => qValues.getOrElse((nextState, a), 0.0)).max
  }

}

object QLearning {
  def apply[E <: Env[IO]](
      env: E,
      nSteps: Int,
      learningRate: Double,
      discountFactor: Double,
      exploration: IO[Exploration[E, IO]],
      logger: BaseLogger[IO]
  ): IO[QLearning[E]] = for {
    qTable <- Ref.of[IO, Map[(E#State, E#Action), Double]](Map.empty)
    buffer <- Ref.of[IO, Queue[(E#State, E#Action, Double)]](Queue.empty)
    explorationActor <- exploration.map {
      case ucb @ UCB(_, _)       => IO.pure(ucb)
      case eg @ EpsilonGreedy(_) => IO.pure(eg)
    }
  } yield new QLearning[E](
    env,
    qTable,
    buffer,
    nSteps,
    learningRate,
    discountFactor,
    explorationActor,
    logger
  )
}
