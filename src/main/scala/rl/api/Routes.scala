package rl.api

import cats.effect.IO
import org.http4s.{EntityDecoder, HttpRoutes}
import org.http4s.dsl.Http4sDsl
import org.http4s.circe._
import io.circe.syntax._
import rl.service.TrainingService

class Routes extends Http4sDsl[IO] {

  implicit val trainRequestDecoder: EntityDecoder[IO, TrainRequest] = jsonOf[IO, TrainRequest]

  val routes: HttpRoutes[IO] = HttpRoutes.of[IO] {
    case GET -> Root / "health" =>
      Ok("OK")

    case req @ POST -> Root / "train" =>
      (for {
        trainReq <- req.as[TrainRequest]
        environment <- ConfigParser.parseEnvironment(trainReq.environment)
        exploration <- ConfigParser.parseExploration(trainReq.exploration)
        agent <- ConfigParser.parseAgent(trainReq.agent, environment, exploration)
        result <- TrainingService.train(agent, trainReq.episodes)
        response <- result.status match {
          case "success" => Ok(TrainResponse(result.message, result.status).asJson)
          case _ => InternalServerError(ErrorResponse(result.message).asJson)
        }
      } yield response).handleErrorWith { err =>
        InternalServerError(ErrorResponse(err.getMessage).asJson)
      }
  }
}
