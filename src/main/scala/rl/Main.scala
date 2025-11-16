package rl

import cats.effect.{IO, IOApp, ExitCode}
import rl.env.GridWorld1D
import rl.agent.QLearning

object Main extends IOApp {

  override def run(args: List[String]): IO[ExitCode] = {
    for {
      env <- GridWorld1D()
      agent <- QLearning(env)
      _ <- IO.println("Reinforcement Learning setup complete.")
      _ <- agent.learn(100)
      _ <- IO.println("Training complete.")
    } yield ExitCode.Success
  }

}
