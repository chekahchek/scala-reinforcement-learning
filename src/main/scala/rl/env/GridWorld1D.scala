package rl.env

import cats.effect.{IO, Ref}
import rl.logging.BaseLogger

/** Environment involves a 1D grid world with 5 locations
  * Action space = {-1, 1} corresponding to {left, right}
  * Observation space = 1D grid of size 5 represented as Int from 0 to 4
  *
  * Example:
  * | 0 - 1 - 2 - 3 - 4 |
  *
  * Start location = 2
  * Target location = 4
  * Goal is to reach the goal location from the start location
  * Rewards = +1.0 for reaching the target location, 0 otherwise
  */

class GridWorld1D(val stateRef: Ref[IO, Int], logger: BaseLogger[IO])
    extends Env[IO] {
  override type Action = Int
  override type State = Int

  // Reset the env in-place by setting the internal Ref to 0 and return this instance
  def reset(): IO[GridWorld1D] = for {
    _ <- stateRef.set(GridWorld1D.initialLocation)
  } yield this

  def step(action: Action): IO[(State, Double, Boolean)] = for {
    currentLocation <- stateRef.get
    newLocation = (currentLocation + action) max 0 min 4
    _ <- stateRef.set(newLocation)
    reward = if (newLocation == 4) 1.0 else 0.0
    done = newLocation == 4
    _ <- logger.debug(
      s"Agent took action: $action moved to location: $newLocation, reward: $reward, done: $done"
    )
  } yield (newLocation, reward, done)

  def getActionSpace: IO[List[Action]] =
    IO.pure(List(-1, 1)) // Move left or right

  def getState: IO[State] = stateRef.get

}

object GridWorld1D {
  private val initialLocation = 2

  def apply(logger: BaseLogger[IO]): IO[GridWorld1D] = for {
    initialLocation <- Ref[IO].of(initialLocation)
  } yield new GridWorld1D(initialLocation, logger)
}
