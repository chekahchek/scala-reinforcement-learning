package rl.env

import cats.effect.{IO, Ref}
import rl.logging.BaseLogger

import scala.annotation.tailrec
import scala.util.Random

/** Environment involves crossing a frozen lake represented as a grid:
  * Action space = {0, 1, 2, 3} corresponding to {left, down, right, up}
  * Observation space = 2D square grid of size 4x4 represented as a Tuple with frozen holes and goal
  *
  * Example grid:
  * | start |       |       |       |
  * |       | hole  |       | hole  |
  * |       |       |       |       |
  * | goal  |       |       |       |
  *
  * Start location is in the top-left corner (0,0)
  * 2 holes that are randomly placed in the grid
  * Goal is to reach the goal location from the start location without falling into any holes.
  * Rewards = +1 for reaching the goal, 0 otherwise
  *
  * Arguments:
  * isSlippery (Boolean) = If true, the agent may slip and move in a different direction than intended. Default to True
  * successRate (Float) = Probability of the agent moving in the intended direction when is_slippery is true. Default to 0.7
  * Example: If agent intend to move left, then it has 70% chance of moving left, and 10% chance each of moving up, down, right.
  *
  * Adapted from https://gymnasium.farama.org/environments/toy_text/frozen_lake/
  */

class FrozenLake(
    val stateRef: Ref[IO, (Int, Int)],
    val isSlippery: Boolean,
    val successRate: Double,
    val goal: (Int, Int),
    val hole1: (Int, Int),
    val hole2: (Int, Int),
    logger: BaseLogger[IO]
) extends Env[IO] {

  override type Action = Int
  override type State = (Int, Int)

  def reset(): IO[FrozenLake] = for {
    _ <- stateRef.set(FrozenLake.initial)
    _ <- logger.debug(
      s"Environment reset. Goal at: $goal, Holes at: $hole1 and $hole2"
    )
  } yield this

  def step(action: Action): IO[(State, Double, Boolean)] = for {
    currentLocation <- stateRef.get
    newAction =
      if (isSlippery && Random.nextDouble() <= successRate) {
        val slipActions = List(0, 1, 2, 3).filter(_ != action)
        slipActions(Random.nextInt(slipActions.length))
      } else {
        action
      }
    newLocation =
      if (newAction % 2 == 0) {
        // left (0) or right (2)
        val newCol =
          (currentLocation._2 + (if (newAction == 0) -1 else 1)) max 0 min 3
        (currentLocation._1, newCol)
      } else {
        // up (3) or down (1)
        val newRow =
          (currentLocation._1 + (if (newAction == 3) -1 else 1)) max 0 min 3
        (newRow, currentLocation._2)
      }

    _ <- stateRef.set(newLocation)
    reward = if (newLocation == goal) 1.0 else 0.0
    done =
      (newLocation == goal) || (newLocation == hole1) || (newLocation == hole2)
    _ <- logger.debug(
      s"Agent took action: $action (with actual action $newAction) moved to location: $newLocation, reward: $reward, done: $done"
    )
  } yield (newLocation, reward, done)

  def getActionSpace: IO[List[Action]] =
    IO.pure(List(0, 1, 2, 3))

  def getState: IO[State] = stateRef.get

}

object FrozenLake {
  private val initial = (0, 0)
  private val minLocation = 0
  private val maxLocation = 3

  @tailrec
  def generateHole(
      minLocation: Int,
      maxLocation: Int,
      excludedLocations: List[(Int, Int)]
  ): (Int, Int) = {
    val row = Random.between(minLocation, maxLocation + 1)
    val col = Random.between(minLocation, maxLocation + 1)
    if (excludedLocations.contains((row, col))) {
      generateHole(minLocation, maxLocation, excludedLocations)
    } else
      (row, col)
  }

  def apply(
      isSlippery: Boolean = true,
      successRate: Double = 0.7,
      logger: BaseLogger[IO]
  ): IO[FrozenLake] = for {
    initialLocation <- Ref[IO].of(initial)

    goal = Random.between(minLocation, maxLocation + 1) -> Random.between(
      minLocation,
      maxLocation + 1
    )
    hole1 = generateHole(minLocation, maxLocation, List(initial, goal))
    hole2 = generateHole(minLocation, maxLocation, List(initial, goal, hole1))

  } yield new FrozenLake(
    initialLocation,
    isSlippery,
    successRate,
    goal,
    hole1,
    hole2,
    logger
  )
}
