package rl.api

import cats.effect.IO
import cats.syntax.functor._
import rl.env.{Env, GridWorld1D, FrozenLake, BlackJack}
import rl.agent.{
  EpsilonGreedy,
  UCB,
  QLearning,
  Sarsa,
  TemporalDifferenceLearning,
  Exploration
}
import rl.logging.{BaseLogger, InfoLogger}

object ConfigParser {

  private val logger: BaseLogger[IO] = InfoLogger

  def parseEnvironment(config: EnvironmentConfig): IO[Env[IO]] = {
    config match {
      case EnvironmentConfig.GridWorld1D =>
        GridWorld1D(logger).widen[Env[IO]]
      case EnvironmentConfig.FrozenLake(isSlippery, successRate) =>
        FrozenLake(isSlippery, successRate, logger).widen[Env[IO]]
      case EnvironmentConfig.BlackJack =>
        BlackJack(logger).widen[Env[IO]]
    }
  }

  def parseExploration(
      config: ExplorationConfig
  ): IO[Exploration[Env[IO], IO]] = {
    config match {
      case ExplorationConfig.EpsilonGreedy(rate) =>
        EpsilonGreedy[Env[IO]](rate).widen[Exploration[Env[IO], IO]]
      case ExplorationConfig.UCB(constant) =>
        UCB[Env[IO]](constant).widen[Exploration[Env[IO], IO]]
    }
  }

  def parseAgent(
      config: AgentConfig,
      env: Env[IO],
      exploration: Exploration[Env[IO], IO]
  ): IO[TemporalDifferenceLearning[Env[IO]]] = {
    config match {
      case AgentConfig.QLearning(lr, df) =>
        QLearning[Env[IO]](env, lr, df, IO.pure(exploration), logger)
      case AgentConfig.Sarsa(lr, df) =>
        Sarsa[Env[IO]](env, lr, df, IO.pure(exploration), logger)
    }
  }
}
