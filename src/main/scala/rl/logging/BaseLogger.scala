package rl.logging

trait BaseLogger[F[_]] {
  def debug(message: String): F[Unit]
  def info(message: String): F[Unit]
  def warn(message: String): F[Unit]
  def error(message: String): F[Unit]
}
