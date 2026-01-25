package rl.agent

import cats.effect.{IO, Ref}
import scala.collection.mutable.Queue
import rl.env.Env
import rl.logging.BaseLogger

abstract class TemporalDifferenceLearning[E <: Env[IO]](
    val env: E,
    protected val qTable: Ref[IO, Map[(E#State, E#Action), Double]],
    protected val buffer: Ref[IO, Queue[(E#State, E#Action, Double)]],
    protected val nSteps: Int,
    protected val learningRate: Double,
    protected val discountFactor: Double,
    protected val explorationActor: IO[Exploration[E, IO]],
    protected val logger: BaseLogger[IO]
) extends Agent[E] {

  def act(state: E#State): IO[E#Action] = for {
    actionSpace <- env.getActionSpace
    qValues <- qTable.get
    action <- explorationActor.flatMap { a =>
      a.getAction(actionSpace, state, qValues)
    }
  } yield action

  /** Peek at what action would be selected without updating exploration state (e.g., UCB counts) */
  def peekAct(state: E#State): IO[E#Action] = for {
    actionSpace <- env.getActionSpace
    qValues <- qTable.get
    action <- explorationActor.flatMap { a =>
      a.peekAction(actionSpace, state, qValues)
    }
  } yield action

// To be overridden by the subclass
  protected def getNextQValue(
      nextState: E#State,
      done: Boolean,
      qValues: Map[(E#State, E#Action), Double],
      actionSpace: List[E#Action]
  ): IO[Double]

  protected def updateQValue(
      done: Boolean,
      nStepsTaken: Int,
      nextState: E#State
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
              qValues <- qTable.get
              currentQ = qValues.getOrElse((firstState, firstAction), 0.0)

              // Add bootstrap value if not done
              // Use the passed nextState (state after last action) for bootstrapping
              bootstrapValue <-
                if (done) IO.pure(0.0)
                else {
                  for {
                    actionSpace <- env.getActionSpace
                    nextQ <- getNextQValue(
                      nextState,
                      done,
                      qValues,
                      actionSpace
                    )
                  } yield Math.pow(discountFactor, rewards.length) * nextQ
                }

              // Update Q-value with n-step return
              updatedQ =
                currentQ + learningRate * (nStepReturn + bootstrapValue - currentQ)
              _ <- qTable.update(qv => qv + ((firstState, firstAction) -> updatedQ))

              // Clear the buffer after update
              _ <- buffer.set(Queue.empty)
            } yield ()
          }
      } yield ()
    } else IO.unit
  }

  def runStep(
      state: E#State,
      action: E#Action
  ): IO[(Boolean, E#State, Double)] = for {
    res <- env.step(action.asInstanceOf[env.Action])
    nextState = res._1.asInstanceOf[E#State]
    reward = res._2
    done = res._3

    // Add transition to buffer
    _ <- buffer.update(_.enqueue((state, action, reward)))
    bufferSize <- buffer.get.map(_.size)

    // Update Q-values when we have n steps or episode is done
    _ <- updateQValue(done, bufferSize, nextState)
  } yield (done, nextState, reward)
}
