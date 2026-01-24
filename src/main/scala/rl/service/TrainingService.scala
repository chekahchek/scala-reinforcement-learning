package rl.service

import cats.effect.IO
import rl.agent.Agent
import rl.env.Env
import rl.logging.{BaseLogger, InfoLogger}
import rl.metrics.TrainingMetrics

case class TrainResult(
    status: String,
    error: Option[String],
    metrics: Option[TrainingMetrics]
)

object TrainResult {
  def success(metrics: TrainingMetrics): TrainResult =
    TrainResult(
      "success",
      None,
      Some(metrics)
    )

  def failure(error: String): TrainResult =
    TrainResult("failed", Some(error), None)
}

object TrainingService {

  private val logger: BaseLogger[IO] = InfoLogger

  def train(
      agent: Agent[Env[IO]],
      episodes: Int
  ): IO[TrainResult] = {
    (for {
      _ <- logger.info(s"Starting training for $episodes episodes")
      metrics <- agent.learn(episodes)
      _ <- logger.info("Training complete")
    } yield TrainResult.success(metrics))
      .handleErrorWith(err => IO.pure(TrainResult.failure(err.getMessage)))
  }
}
