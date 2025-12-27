package rl

import cats.effect.{ExitCode, IO, IOApp}
import org.http4s.blaze.server.BlazeServerBuilder
import org.http4s.implicits._
import rl.api.Routes

object Main extends IOApp {

  override def run(args: List[String]): IO[ExitCode] = {
    val routes = new Routes().routes

    IO.println("Starting server on http://0.0.0.0:8080 ...") *>
      BlazeServerBuilder[IO]
        .bindHttp(8080, "0.0.0.0")
        .withHttpApp(routes.orNotFound)
        .serve
        .compile
        .drain
        .as(ExitCode.Success)
  }

}
