package rl.env

import cats.effect.{IO, Ref}

/**
 *
 * @param stateRef Ref storing the state of the 1D Grid World
 *                 | 0 - 1 - 2 - 3 - 4 |
 *                 Initial location = 2
 *                 Target location = 4
 *                 Reward = 1.0 at target location, else 0.0
 */
class GridWorld1D(val stateRef: Ref[IO, Int]) extends Env[IO] {
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
    reward = if(newLocation == 4) 1.0 else 0.0
    done = newLocation == 4
  } yield (newLocation, reward, done)

  def getActionSpace: IO[List[Action]] = IO.pure(List(-1, 1)) // Move left or right

  def getState: IO[State] = stateRef.get

  def renderState(state: State): IO[String] = for {
    location <- stateRef.get
  } yield location.toString
}

object GridWorld1D {
  private val initialLocation = 2

  def apply(): IO[GridWorld1D] = for {
    initialLocation <- Ref[IO].of(initialLocation)
  } yield new GridWorld1D(initialLocation)
}