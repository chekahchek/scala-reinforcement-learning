package rl.service

import cats.effect.IO
import rl.agent.Agent
import rl.env.Env
import rl.logging.{BaseLogger, InfoLogger}

case class TrainResult(
    episodes: Int,
    status: String,
    message: String
)

object TrainResult {
  def success(episodes: Int): TrainResult =
    TrainResult(
      episodes,
      "success",
      s"Training completed successfully for $episodes episodes"
    )

  def failure(episodes: Int, error: String): TrainResult =
    TrainResult(episodes, "failed", s"Training failed: $error")
}

object TrainingService {

  private val logger: BaseLogger[IO] = InfoLogger

  def train(
      agent: Agent[Env[IO], IO],
      episodes: Int
  ): IO[TrainResult] = {
    (for {
      _ <- logger.info(s"Starting training for $episodes episodes")
      _ <- agent.learn(episodes)
      _ <- logger.info("Training complete")
    } yield TrainResult.success(episodes))
      .handleError(err => TrainResult.failure(episodes, err.getMessage))
  }
}
