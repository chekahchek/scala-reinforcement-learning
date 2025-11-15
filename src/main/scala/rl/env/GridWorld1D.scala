package rl.env

import cats.effect.{IO, Ref}

case class GridWorld1DState(stateRef: Ref[IO, Int])

class GridWorld1D(val state: GridWorld1DState) extends Env[IO]{
  override type Action = Int
  override type State = GridWorld1DState

  // Reset the env in-place by setting the internal Ref to 0 and return this instance
  def reset(): IO[GridWorld1D] = for {
    _ <- state.stateRef.set(0)
  } yield this

  def step(action: Action): IO[(GridWorld1DState, Double, Boolean)] = for {
    currentLocation <- state.stateRef.get
    newLocation = (currentLocation + action) max 0 min 4
    _ <- state.stateRef.set(newLocation)
    reward = if (newLocation == 4) 1.0 else 0.0
    done = newLocation == 4
  } yield (state, reward, done)

  def getActionSpace: IO[List[Action]] = IO.pure(List(-1, 1)) // Move left or right
  def getState: IO[GridWorld1DState] = IO.pure(state)
}

object GridWorld1D {
  def apply() : IO[GridWorld1D] = for {
    initialLocation <- Ref[IO].of(0)
    state = GridWorld1DState(initialLocation)
  } yield new GridWorld1D(state)
}