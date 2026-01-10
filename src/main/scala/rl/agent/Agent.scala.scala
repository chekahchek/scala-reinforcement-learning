package rl.agent

import cats.effect.IO
import rl.env.Env
import rl.metrics.TrainingMetrics

trait Agent[E <: Env[F], F[_]] {
  def learn(episodes: Int): F[TrainingMetrics]
}
