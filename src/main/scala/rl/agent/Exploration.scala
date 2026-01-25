package rl.agent

import cats.effect.{IO, Ref}
import rl.env.Env

import scala.util.Random
import rl.logging.BaseLogger

trait Exploration[E <: Env[F], F[_]] {
  def getAction(
      actionSpace: List[E#Action],
      state: E#State,
      qValues: Map[(E#State, E#Action), Double]
  ): F[E#Action]

  /** Peek at what action would be selected without updating internal state (e.g., UCB counts) */
  def peekAction(
      actionSpace: List[E#Action],
      state: E#State,
      qValues: Map[(E#State, E#Action), Double]
  ): F[E#Action] = getAction(actionSpace, state, qValues)
}

case class EpsilonGreedy[E <: Env[IO]](explorationRate: Double)
    extends Exploration[E, IO] {
  override def getAction(
      actionSpace: List[E#Action],
      state: E#State,
      qValues: Map[(E#State, E#Action), Double]
  ): IO[E#Action] = for {
    action <-
      if (Random.nextDouble() < explorationRate) {
        // Explore: choose a random action
        IO.pure(actionSpace(Random.nextInt(actionSpace.length)))
      } else {
        // Exploit: choose the best action based on Q-values
        val bestAction =
          actionSpace.maxBy(action => qValues.getOrElse((state, action), 0.0))
        IO.pure(bestAction)
      }
  } yield action
}

case class UCB[E <: Env[IO]](
    constant: Int,
    ucbRef: Ref[IO, Map[(E#State, E#Action), Int]],
) extends Exploration[E, IO] {

  private def selectAction(
      actionSpace: List[E#Action],
      state: E#State,
      qValues: Map[(E#State, E#Action), Double],
      stateActionCount: Map[(E#State, E#Action), Int]
  ): E#Action = {

    // UCB Formula: Argmax_a (qValue + constant * sqrt(log(totalCounts) / actionCount))
    // totalCounts = Sum of all the counts for all actions for the given state
    // actionCount = Count of the specific action in the given state
    val totalCounts = actionSpace
      .map(a => stateActionCount.getOrElse((state, a), 0))
      .sum

    actionSpace.maxBy { a =>
      val qValue = qValues.getOrElse((state, a), 0.0)
      val actionCount = stateActionCount.getOrElse((state, a), 0)
      if (actionCount == 0) Double.MaxValue
      else
        qValue + constant * Math.sqrt(
          Math.log(totalCounts.toDouble) / actionCount.toDouble
        )
    }
  }

  override def getAction(
      actionSpace: List[E#Action],
      state: E#State,
      qValues: Map[(E#State, E#Action), Double]
  ): IO[E#Action] = for {
    stateActionCount <- ucbRef.get
    action = selectAction(actionSpace, state, qValues, stateActionCount)
    // Update the count for the selected action
    _ <- ucbRef.update { counts =>
      val currentCount = counts.getOrElse((state, action), 0)
      counts + ((state, action) -> (currentCount + 1))
    }
  } yield action

  /** Peek at what action would be selected without updating counts */
  override def peekAction(
      actionSpace: List[E#Action],
      state: E#State,
      qValues: Map[(E#State, E#Action), Double]
  ): IO[E#Action] = for {
    stateActionCount <- ucbRef.get
    action = selectAction(actionSpace, state, qValues, stateActionCount)
  } yield action
}

object EpsilonGreedy {
  def apply[E <: Env[IO]](explorationRate: Double): IO[EpsilonGreedy[E]] =
    IO.pure(new EpsilonGreedy[E](explorationRate))
}

object UCB {
  def apply[E <: Env[IO]](constant: Int): IO[UCB[E]] = for {
    ucbRef <- Ref[IO].of(Map.empty[(E#State, E#Action), Int])
  } yield UCB[E](constant, ucbRef)
}
