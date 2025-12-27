package rl.api

import cats.effect.IO
import org.http4s.{EntityDecoder, HttpRoutes}
import org.http4s.dsl.Http4sDsl
import org.http4s.circe._
import io.circe.syntax._

class Routes extends Http4sDsl[IO] {

  implicit val trainRequestDecoder: EntityDecoder[IO, TrainRequest] = jsonOf[IO, TrainRequest]

  val routes: HttpRoutes[IO] = HttpRoutes.of[IO] {
    case GET -> Root / "health" =>
      Ok("OK")

    case req @ POST -> Root / "train" =>
      (for {
        trainReq <- req.as[TrainRequest]
        environment <- TrainService.parseEnvironment(trainReq.environment)
        exploration <- TrainService.parseExploration(trainReq.exploration)
        agent <- TrainService.parseAgent(trainReq.agent, environment, exploration)
        response <- TrainService.train(agent, trainReq.episodes)
      } yield response).handleErrorWith { err =>
        InternalServerError(ErrorResponse(err.getMessage).asJson)
      }
  }
}
