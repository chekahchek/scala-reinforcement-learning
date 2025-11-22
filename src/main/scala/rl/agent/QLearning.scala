package rl.agent

import cats.effect.{IO, Ref}
import rl.env.Env
import scala.util.Random
import rl.logging.BaseLogger

class QLearning[E <: Env[IO]](val env: E,
                              qTable: Ref[IO, Map[(E#State, E#Action), Double]],
                              learningRate: Double,
                              discountFactor: Double,
                              exploration: Exploration.Exploration,
                              stateActionCount: Ref[IO, Map[(E#State, E#Action), Int]],
                              logger: BaseLogger[IO]
                             ) {

  def epsilonGreedyAction(actionSpace: List[env.Action], state: env.State, explorationRate: Double): IO[env.Action] = for {
    qValues <- qTable.get
    action <- if (Random.nextDouble() < explorationRate) {
      // Explore: choose a random action
      IO.pure(actionSpace(Random.nextInt(actionSpace.length)))
    } else {
      // Exploit: choose the best action based on Q-values
      val bestAction = actionSpace.maxBy(action => qValues.getOrElse((state, action), 0.0))
      IO.pure(bestAction)
    }
  } yield action

  def ucbAction(actionSpace: List[env.Action], state: env.State, constant: Int): IO[env.Action] = for {
    qValues <- qTable.get
    counts <- stateActionCount.get
    totalCounts = actionSpace.map(a => counts.getOrElse((state, a), 0)).sum
    action <- IO.pure {
      actionSpace.maxBy { a =>
        val qValue = qValues.getOrElse((state, a), 0.0)
        val actionCount = counts.getOrElse((state, a), 0)
        if (actionCount == 0) Double.MaxValue
        else qValue + constant * Math.sqrt(Math.log(totalCounts.toDouble) / actionCount.toDouble)
      }
    }
    // Update the count for the selected action
    _ <- stateActionCount.update { sc =>
      val currentCount = sc.getOrElse((state, action), 0)
      sc + ((state, action) -> (currentCount + 1))
    }
  } yield action


  def act(state: env.State): IO[env.Action] = for {
    actionSpace <- env.getActionSpace
    action <- exploration match {
      case Exploration.EpsilonGreedy(explorationRate) =>
        epsilonGreedyAction(actionSpace, state, explorationRate)
      case Exploration.UCB(constant) =>
        ucbAction(actionSpace, state, constant)
    }
  } yield action

  def runStep(state: env.State): IO[Boolean] = for {
    action <- act(state)
    res <- env.step(action)
    nextState = res._1
    reward = res._2
    done = res._3
    nextStateStr <- env.renderState(nextState)
    _ <- logger.debug(s"Agent took action: $action resulting in state: $nextStateStr, reward: ${res._2}, done: ${res._3}")

    qValues <- qTable.get
    actionSpace <- env.getActionSpace

    // Update Q-values
    currentQ = qValues.getOrElse((state, action), 0.0)
    maxNextQ = if (done) 0.0 else actionSpace.map(a => qValues.getOrElse((nextState, a), 0.0)).max
    updatedQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ)
    _ <- qTable.update(qv => qv + ((state, action) -> updatedQ))
  } yield done

  def runEpisode(): IO[Unit] = {
    def loop(state: env.State): IO[Unit] = for {
      done <- runStep(state)
      _ <- if (done) IO.unit
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
      if (episode >= numEpisodes) IO.unit
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
                          exploration: Exploration.Exploration,
                          logger: BaseLogger[IO]
                         ): IO[QLearning[E]] = for {
    qTable <- Ref.of[IO, Map[(E#State, E#Action), Double]](Map.empty)
    stateActionCount <- Ref.of[IO, Map[(E#State, E#Action), Int]](Map.empty)
  } yield new QLearning[E](env, qTable, learningRate, discountFactor, exploration, stateActionCount, logger)
}
