package rl.agent

import cats.effect.IO
import rl.env.Env

trait Agent[E <: Env[F], F[_]] {
  def learn(episodes: Int): F[Unit]
}
