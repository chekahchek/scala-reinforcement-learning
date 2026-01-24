package rl.agent

import cats.effect.{IO, Ref}
import cats.effect.std.Random
import scala.collection.mutable.Queue
import rl.env.Env
import rl.logging.BaseLogger

class DoubleQLearning[E <: Env[IO]](
    val env: E,
    qTable1: Ref[IO, Map[(E#State, E#Action), Double]],
    qTable2: Ref[IO, Map[(E#State, E#Action), Double]],
    buffer: Ref[IO, Queue[(E#State, E#Action, Double)]],
    nSteps: Int,
    learningRate: Double,
    discountFactor: Double,
    explorationActor: IO[Exploration[E, IO]],
    protected val logger: BaseLogger[IO],
    random: Random[IO]
) extends Agent[E] {

  def act(state: E#State): IO[E#Action] = act(state, qTable1)

  private def act(
      state: E#State,
      selectActionTable: Ref[IO, Map[(E#State, E#Action), Double]]
  ): IO[E#Action] = for {
    actionSpace <- env.getActionSpace
    qValues <- selectActionTable.get
    action <- explorationActor.flatMap { a =>
      a.getAction(actionSpace, state, qValues)
    }
  } yield action

  def getNextQValue(
      nextState: E#State,
      done: Boolean,
      selectQValues: Map[(E#State, E#Action), Double],
      evalQValues: Map[(E#State, E#Action), Double],
      actionSpace: List[E#Action]
  ): IO[Double] = IO.pure {
    if (done) 0.0
    else {
      // Assuming Q1 = SelectActionTable, we find out best action using Q1: a* = argmax Q1(s',a')
      // Then we use Q2 to evaluate the action: Q2(s', a*)
      val bestAction =
        actionSpace.maxBy(a => selectQValues.getOrElse((nextState, a), 0.0))
      evalQValues.getOrElse((nextState, bestAction), 0.0)
    }
  }

  def updateQValue(
      done: Boolean,
      nStepsTaken: Int,
      nextState: E#State,
      selectActionTable: Ref[IO, Map[(E#State, E#Action), Double]],
      evalActionTable: Ref[IO, Map[(E#State, E#Action), Double]]
  ): IO[Unit] = {
    if (done || nStepsTaken >= nSteps) {
      for {
        bufferQueue <- buffer.get
        transitions = bufferQueue.dequeueAll(_ => true)
        _ <-
          if (transitions.isEmpty) IO.unit
          else {
            val (states, actions, rewards) = transitions.unzip3
            // Calculate n-step return: sum of discounted rewards
            val nStepReturn = rewards.zipWithIndex.foldLeft(0.0) { case (acc, (reward, index)) =>
              acc + reward * Math.pow(discountFactor, index)
            }

            // Get the first state-action pair
            val firstState = states.head
            val firstAction = actions.head

            for {
              selectQValues <- selectActionTable.get
              evalQValues <- evalActionTable.get

              // Update Q-value based on the table where action was selected
              currentQ = selectQValues.getOrElse((firstState, firstAction), 0.0)

              // Add bootstrap value if not done
              bootstrapValue <-
                if (done) IO.pure(0.0)
                else {
                  for {
                    actionSpace <- env.getActionSpace

                    nextQ <- getNextQValue(
                      nextState,
                      done,
                      selectQValues,
                      evalQValues,
                      actionSpace
                    )
                  } yield Math.pow(discountFactor, rewards.length) * nextQ
                }

              // Update Q-value with n-step return based on the table where action was selected
              updatedQ =
                currentQ + learningRate * (nStepReturn + bootstrapValue - currentQ)
              _ <- selectActionTable.update(qv => qv + ((firstState, firstAction) -> updatedQ))

              // Clear the buffer after update
              _ <- buffer.set(Queue.empty)
            } yield ()
          }
      } yield ()
    } else IO.unit
  }

  protected def runStep(
      state: E#State,
      action: E#Action
  ): IO[(Boolean, E#State, Double)] = runStep(state, action, qTable1, qTable2)

  private def runStep(
      state: E#State,
      action: E#Action,
      selectActionTable: Ref[IO, Map[(E#State, E#Action), Double]],
      evalActionTable: Ref[IO, Map[(E#State, E#Action), Double]]
  ): IO[(Boolean, E#State, Double)] = for {
    res <- env.step(action.asInstanceOf[env.Action])
    nextState = res._1.asInstanceOf[E#State]
    reward = res._2
    done = res._3

    // Add transition to buffer
    _ <- buffer.update(_.enqueue((state, action, reward)))
    bufferSize <- buffer.get.map(_.size)

    // Update Q-values when we have n steps or episode is done
    _ <- updateQValue(
      done,
      bufferSize,
      nextState,
      selectActionTable,
      evalActionTable
    )
  } yield (done, nextState, reward)

  override def runEpisode(): IO[(Double, Int)] = {
    def loop(
        state: E#State,
        reward: Double,
        stepCount: Int
    ): IO[(Double, Int)] = for {
      coinFlip <- random.nextBoolean
      tables = if (coinFlip) (qTable1, qTable2) else (qTable2, qTable1)
      (selectActionTable, evalActionTable) = tables
      action <- act(state, selectActionTable)
      result <- runStep(state, action, selectActionTable, evalActionTable)
      (done, nextState, rewardObtained) = result
      totalEpisodeReward = reward + rewardObtained
      totalStepCount = stepCount + 1
      result <-
        if (done) IO.pure((totalEpisodeReward, totalStepCount))
        else loop(nextState, totalEpisodeReward, totalStepCount)
    } yield result

    for {
      _ <- env.reset()
      initialState <- env.getState
      initialReward = 0.0
      initialStepCount = 0
      episodeResult <- loop(initialState, initialReward, initialStepCount)
    } yield episodeResult
  }
}

object DoubleQLearning {
  def apply[E <: Env[IO]](
      env: E,
      nSteps: Int,
      learningRate: Double,
      discountFactor: Double,
      exploration: IO[Exploration[E, IO]],
      logger: BaseLogger[IO]
  ): IO[DoubleQLearning[E]] = for {
    qTable1 <- Ref.of[IO, Map[(E#State, E#Action), Double]](Map.empty)
    qTable2 <- Ref.of[IO, Map[(E#State, E#Action), Double]](Map.empty)
    buffer <- Ref.of[IO, Queue[(E#State, E#Action, Double)]](Queue.empty)
    random <- Random.scalaUtilRandom[IO]
    explorationActor <- exploration.map {
      case ucb @ UCB(_, _)       => IO.pure(ucb)
      case eg @ EpsilonGreedy(_) => IO.pure(eg)
    }
  } yield new DoubleQLearning[E](
    env,
    qTable1,
    qTable2,
    buffer,
    nSteps,
    learningRate,
    discountFactor,
    explorationActor,
    logger,
    random
  )
}
