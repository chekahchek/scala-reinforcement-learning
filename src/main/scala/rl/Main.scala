package rl

import cats.effect.unsafe.implicits.global
import cats.effect.{ExitCode, IO, IOApp, Ref}
import rl.env.GridWorld1D
import rl.agent.{EpsilonGreedy, QLearning, UCB}
import rl.logging.{DebugLogger, InfoLogger, NoOpLogger}

object Main extends IOApp {

  override def run(args: List[String]): IO[ExitCode] = {
    import cats.syntax.all._
    for {
      env <- GridWorld1D()
      logger = DebugLogger
      explorationMethod = EpsilonGreedy[GridWorld1D](explorationRate = 0.1)
      // TODO: Simplify this
//      explorationMethod = {
//        val ref = Ref[IO].of(Map.empty[(env.State, env.Action), Int])
//        UCB[GridWorld1D](constant = 2, ucbRef = ref.unsafeRunSync())
//      }
      agent <- QLearning(env, exploration=explorationMethod, logger=logger)
      _ <- IO.println("Reinforcement Learning setup complete.")
      _ <- agent.learn(100)
      _ <- IO.println("Training complete.")
    }

yield ExitCode.Success
  }

}
