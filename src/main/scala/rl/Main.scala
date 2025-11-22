package rl

import cats.effect.{IO, IOApp, ExitCode}
import rl.env.GridWorld1D
import rl.agent.{QLearning, Exploration}
import rl.logging.{DebugLogger, InfoLogger, NoOpLogger}

object Main extends IOApp {

  override def run(args: List[String]): IO[ExitCode] = {
    for {
      env <- GridWorld1D()
      logger = DebugLogger
      explorationMethod = Exploration.UCB(10)
      agent <- QLearning(env, exploration=explorationMethod, logger=logger)
      _ <- IO.println("Reinforcement Learning setup complete.")
      _ <- agent.learn(100)
      _ <- IO.println("Training complete.")
    }

yield ExitCode.Success
  }

}
