package rl.api

import io.circe.{Decoder, Encoder}
import io.circe.generic.semiauto._
import rl.metrics.{EpisodeMetrics, TrainingMetrics}

// Request models
case class TrainRequest(
    environment: EnvironmentConfig,
    agent: AgentConfig,
    exploration: ExplorationConfig,
    episodes: Int
)

sealed trait EnvironmentConfig
object EnvironmentConfig {
  case object GridWorld1D extends EnvironmentConfig
  case class FrozenLake(isSlippery: Boolean = true, successRate: Double = 0.7) extends EnvironmentConfig
  case object BlackJack extends EnvironmentConfig

  implicit val decoder: Decoder[EnvironmentConfig] = Decoder.instance { cursor =>
    cursor.downField("type").as[String].flatMap {
      case "GridWorld1D" => Right(GridWorld1D)
      case "BlackJack" => Right(BlackJack)
      case "FrozenLake" =>
        for {
          isSlippery <- cursor.downField("isSlippery").as[Option[Boolean]].map(_.getOrElse(true))
          successRate <- cursor.downField("successRate").as[Option[Double]].map(_.getOrElse(0.7))
        } yield FrozenLake(isSlippery, successRate)
      case other => Left(io.circe.DecodingFailure(s"Unknown environment type: $other", cursor.history))
    }
  }

  implicit val encoder: Encoder[EnvironmentConfig] = Encoder.instance {
    case GridWorld1D => io.circe.Json.obj("type" -> io.circe.Json.fromString("GridWorld1D"))
    case BlackJack => io.circe.Json.obj("type" -> io.circe.Json.fromString("BlackJack"))
    case FrozenLake(isSlippery, successRate) =>
      io.circe.Json.obj(
        "type" -> io.circe.Json.fromString("FrozenLake"),
        "isSlippery" -> io.circe.Json.fromBoolean(isSlippery),
        "successRate" -> io.circe.Json.fromDouble(successRate).getOrElse(io.circe.Json.fromDoubleOrNull(successRate))
      )
  }
}

sealed trait AgentConfig
object AgentConfig {
  case class QLearning(
      learningRate: Double = 0.1,
      discountFactor: Double = 0.9,
      nSteps: Int = 1
  ) extends AgentConfig
  case class Sarsa(
      learningRate: Double = 0.1,
      discountFactor: Double = 0.9,
      nSteps: Int = 1
  ) extends AgentConfig
  case class DoubleQLearning(
      learningRate: Double = 0.1,
      discountFactor: Double = 0.9,
      nSteps: Int = 1
  ) extends AgentConfig

  implicit val decoder: Decoder[AgentConfig] = Decoder.instance { cursor =>
    cursor.downField("type").as[String].flatMap {
      case "QLearning" =>
        for {
          lr <- cursor.downField("learningRate").as[Option[Double]].map(_.getOrElse(0.1))
          df <- cursor.downField("discountFactor").as[Option[Double]].map(_.getOrElse(0.9))
          nSteps <- cursor.downField("nSteps").as[Option[Int]].map(_.getOrElse(1))
        } yield QLearning(lr, df, nSteps)
      case "Sarsa" =>
        for {
          lr <- cursor.downField("learningRate").as[Option[Double]].map(_.getOrElse(0.1))
          df <- cursor.downField("discountFactor").as[Option[Double]].map(_.getOrElse(0.9))
          nSteps <- cursor.downField("nSteps").as[Option[Int]].map(_.getOrElse(1))
        } yield Sarsa(lr, df, nSteps)
      case "DoubleQLearning" =>
        for {
          lr <- cursor.downField("learningRate").as[Option[Double]].map(_.getOrElse(0.1))
          df <- cursor.downField("discountFactor").as[Option[Double]].map(_.getOrElse(0.9))
          nSteps <- cursor.downField("nSteps").as[Option[Int]].map(_.getOrElse(1))
        } yield DoubleQLearning(lr, df, nSteps)
      case other => Left(io.circe.DecodingFailure(s"Unknown agent type: $other", cursor.history))
    }
  }

  implicit val encoder: Encoder[AgentConfig] = Encoder.instance {
    case QLearning(lr, df, nSteps) =>
      io.circe.Json.obj(
        "type" -> io.circe.Json.fromString("QLearning"),
        "learningRate" -> io.circe.Json.fromDouble(lr).getOrElse(io.circe.Json.fromDoubleOrNull(lr)),
        "discountFactor" -> io.circe.Json.fromDouble(df).getOrElse(io.circe.Json.fromDoubleOrNull(df)),
        "nSteps" -> io.circe.Json.fromInt(nSteps)
      )
    case Sarsa(lr, df, nSteps) =>
      io.circe.Json.obj(
        "type" -> io.circe.Json.fromString("Sarsa"),
        "learningRate" -> io.circe.Json.fromDouble(lr).getOrElse(io.circe.Json.fromDoubleOrNull(lr)),
        "discountFactor" -> io.circe.Json.fromDouble(df).getOrElse(io.circe.Json.fromDoubleOrNull(df)),
        "nSteps" -> io.circe.Json.fromInt(nSteps)
      )
    case DoubleQLearning(lr, df, nSteps) =>
      io.circe.Json.obj(
        "type" -> io.circe.Json.fromString("DoubleQLearning"),
        "learningRate" -> io.circe.Json.fromDouble(lr).getOrElse(io.circe.Json.fromDoubleOrNull(lr)),
        "discountFactor" -> io.circe.Json.fromDouble(df).getOrElse(io.circe.Json.fromDoubleOrNull(df)),
        "nSteps" -> io.circe.Json.fromInt(nSteps)
      )
  }
}

sealed trait ExplorationConfig
object ExplorationConfig {
  case class EpsilonGreedy(explorationRate: Double = 0.1) extends ExplorationConfig
  case class UCB(constant: Int = 1) extends ExplorationConfig

  implicit val decoder: Decoder[ExplorationConfig] = Decoder.instance { cursor =>
    cursor.downField("type").as[String].flatMap {
      case "EpsilonGreedy" =>
        cursor.downField("explorationRate").as[Option[Double]]
          .map(rate => EpsilonGreedy(rate.getOrElse(0.1)))
      case "UCB" =>
        cursor.downField("constant").as[Option[Int]]
          .map(c => UCB(c.getOrElse(1)))
      case other => Left(io.circe.DecodingFailure(s"Unknown exploration type: $other", cursor.history))
    }
  }

  implicit val encoder: Encoder[ExplorationConfig] = Encoder.instance {
    case EpsilonGreedy(rate) =>
      io.circe.Json.obj(
        "type" -> io.circe.Json.fromString("EpsilonGreedy"),
        "explorationRate" -> io.circe.Json.fromDouble(rate).getOrElse(io.circe.Json.fromDoubleOrNull(rate))
      )
    case UCB(constant) =>
      io.circe.Json.obj(
        "type" -> io.circe.Json.fromString("UCB"),
        "constant" -> io.circe.Json.fromInt(constant)
      )
  }
}

// Response models
case class EpisodeMetricsResponse(
    episodeNumber: Int,
    totalReward: Double,
    totalSteps: Int
)

object EpisodeMetricsResponse {
  implicit val encoder: Encoder[EpisodeMetricsResponse] = deriveEncoder[EpisodeMetricsResponse]
  implicit val decoder: Decoder[EpisodeMetricsResponse] = deriveDecoder[EpisodeMetricsResponse]
  
  def fromEpisodeMetrics(em: EpisodeMetrics): EpisodeMetricsResponse =
    EpisodeMetricsResponse(em.episodeNumber, em.totalReward, em.totalSteps)
}

case class TrainingMetricsResponse(
    episodeMetrics: List[EpisodeMetricsResponse]
)

object TrainingMetricsResponse {
  implicit val encoder: Encoder[TrainingMetricsResponse] = deriveEncoder[TrainingMetricsResponse]
  implicit val decoder: Decoder[TrainingMetricsResponse] = deriveDecoder[TrainingMetricsResponse]
  
  def fromTrainingMetrics(tm: TrainingMetrics): TrainingMetricsResponse =
    TrainingMetricsResponse(tm.episodeMetrics.map(EpisodeMetricsResponse.fromEpisodeMetrics))
}

case class TrainResponse(
    status: String,
    error: Option[String],
    metrics: Option[TrainingMetricsResponse]
)

object TrainResponse {
  implicit val encoder: Encoder[TrainResponse] = deriveEncoder[TrainResponse]
  implicit val decoder: Decoder[TrainResponse] = deriveDecoder[TrainResponse]
}

case class ErrorResponse(
    error: String
)

object ErrorResponse {
  implicit val encoder: Encoder[ErrorResponse] = deriveEncoder[ErrorResponse]
  implicit val decoder: Decoder[ErrorResponse] = deriveDecoder[ErrorResponse]
}

object TrainRequest {
  implicit val decoder: Decoder[TrainRequest] = deriveDecoder[TrainRequest]
  implicit val encoder: Encoder[TrainRequest] = deriveEncoder[TrainRequest]
}

