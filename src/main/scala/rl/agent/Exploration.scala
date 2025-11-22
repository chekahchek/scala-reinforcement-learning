package rl.agent

object Exploration {
  sealed trait Exploration
  case class EpsilonGreedy(explorationRate: Double) extends Exploration
  case class UCB(constant: Int) extends Exploration
}




