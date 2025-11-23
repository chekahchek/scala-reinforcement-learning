package rl

import cats.effect.{ExitCode, IO, IOApp}
import rl.env.GridWorld1D
import rl.agent.{EpsilonGreedy, QLearning, UCB}
import rl.logging.{DebugLogger, InfoLogger, NoOpLogger}

object Main extends IOApp {

  override def run(args: List[String]): IO[ExitCode] = {
    for {
      env <- GridWorld1D()
      logger = DebugLogger
      explorationMethod = UCB[GridWorld1D](constant = 1)
//      explorationMethod = EpsilonGreedy[GridWorld1D](explorationRate = 0.1)
      agent <- QLearning(env, exploration=explorationMethod, logger=logger)
      _ <- IO.println("Reinforcement Learning setup complete.")
      _ <- agent.learn(100)
      _ <- IO.println("Training complete.")
    }

yield ExitCode.Success
  }

}
