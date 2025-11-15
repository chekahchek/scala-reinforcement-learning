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
      _ <- trainEpisode(agent, maxSteps = 100)
    } yield ExitCode.Success
  }

  def trainEpisode(agentInstance: QLearning[GridWorld1D], maxSteps: Int): IO[Unit] = {
    val environment = agentInstance.env

    // Use the environment's State type for the loop parameter
    def loop(state: environment.State, stepCount: Int): IO[Unit] = {
      if (stepCount >= maxSteps) IO.unit
      else for {
        action <- agentInstance.act(state)
        res <- environment.step(action)
        nextState = res._1
        reward = res._2
        done = res._3
        _ <- agentInstance.learn(state, action, reward, nextState)
        _ <- if (done) IO.unit else loop(nextState, stepCount + 1)
      } yield ()
    }
    for {
      // reset using the environment's reset method
      resetEnv <- environment.reset()
      initialState <- IO.pure(resetEnv.state)
      _ <- loop(initialState, 0)
    } yield ()
  }

}
