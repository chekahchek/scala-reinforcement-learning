package rl.agent

import cats.effect.{IO, Ref}
import rl.env.Env
import scala.util.Random

class QLearning[E <: Env[IO]](val env: E,
                qTable: Ref[IO, Map[Any, Double]],
                learningRate: Double,
                discountFactor: Double,
                explorationRate: Double) {

  def act(state: env.State): IO[env.Action] = for {
    actionSpace <- env.getActionSpace
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

  def runStep(state: env.State): IO[Boolean] = for {
    action <- act(state)
    res <- env.step(action)
    nextState = res._1
    reward = res._2
    done = res._3
    nextStateStr <- env.renderState(nextState)
    _ <- IO.println(s"Agent took action: $action resulting in state: $nextStateStr, reward: ${res._2}, done: ${res._3}")

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
        _ <- IO.println(s"Completed episode: ${episode + 1}")
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
                          explorationRate: Double = 0.1): IO[QLearning[E]] = for {
    qTable <- Ref.of[IO, Map[Any, Double]](Map.empty)
  } yield new QLearning[E](env, qTable, learningRate, discountFactor, explorationRate)
}
