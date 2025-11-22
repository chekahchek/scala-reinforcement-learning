package rl

import cats.effect.{IO, IOApp, ExitCode}
import rl.env.GridWorld1D
import rl.agent.QLearning
import rl.logging.{DebugLogger, InfoLogger, NoOpLogger}

object Main extends IOApp {

  override def run(args: List[String]): IO[ExitCode] = {
    for {
      env <- GridWorld1D()
      logger = DebugLogger
      agent <- QLearning(env, logger=logger)
      _ <- IO.println("Reinforcement Learning setup complete.")
      _ <- agent.learn(100)
      _ <- IO.println("Training complete.")
    }

yield ExitCode.Success
  }

}
