package rl.logging

import cats.effect.IO

object DebugLogger extends BaseLogger[IO] {
  def debug(message: String): IO[Unit] = IO.println(s"[DEBUG] $message")

  def info(message: String): IO[Unit] = IO.println(s"[INFO] $message")

  def warn(message: String): IO[Unit] = IO.println(s"[WARN] $message")

  def error(message: String): IO[Unit] = IO.println(s"[ERROR] $message")
}

object NoOpLogger extends BaseLogger[IO] {
  def debug(message: String): IO[Unit] = IO.unit

  def info(message: String): IO[Unit] = IO.unit

  def warn(message: String): IO[Unit] = IO.unit

  def error(message: String): IO[Unit] = IO.unit
}

object InfoLogger extends BaseLogger[IO] {
  def debug(message: String): IO[Unit] = IO.unit

  def info(message: String): IO[Unit] = IO.println(s"[INFO] $message")

  def warn(message: String): IO[Unit] = IO.println(s"[WARN] $message")

  def error(message: String): IO[Unit] = IO.println(s"[ERROR] $message")
}
