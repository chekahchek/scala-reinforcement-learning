package rl

import cats.effect.{IO, Ref}
import scala.collection.mutable.Queue
import rl.env.GridWorld1D
import rl.agent.{Exploration, EpsilonGreedy}
import rl.logging.{BaseLogger, NoOpLogger}

object TestFixture {

  // Fixture for testing agents
  case class AgentTestFixture(
      env: GridWorld1D,
      qTable: Ref[IO, Map[(Int, Int), Double]],
      buffer: Ref[IO, Queue[(Int, Int, Double)]],
      explorationActor: IO[Exploration[GridWorld1D, IO]],
      learningRate: Double,
      discountFactor: Double,
      logger: BaseLogger[IO]
  )

  object AgentTestFixture {
    def apply(
        initialQValues: Map[(Int, Int), Double] = Map.empty,
        explorationRate: Double = 0.0,
        learningRate: Double = 0.1,
        discountFactor: Double = 0.99
    ): IO[AgentTestFixture] = for {
      env <- GridWorld1D(NoOpLogger)
      qTable <- Ref.of[IO, Map[(Int, Int), Double]](initialQValues)
      buffer <- Ref.of[IO, Queue[(Int, Int, Double)]](Queue.empty)
      explorationActor = IO.pure(new EpsilonGreedy[GridWorld1D](explorationRate))
    } yield AgentTestFixture(
      env,
      qTable,
      buffer,
      explorationActor,
      learningRate,
      discountFactor,
      NoOpLogger
    )
  }
}
