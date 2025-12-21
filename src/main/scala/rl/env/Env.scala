package rl.env

trait Env[F[_]] {
  type Action
  type ActionSpace = List[Action]
  type State

  def reset(): F[Env[F]]

  def step(action: Action): F[(State, Double, Boolean)] // State, Reward, Done

  def getActionSpace: F[List[Action]]

  def getState: F[State]
}
