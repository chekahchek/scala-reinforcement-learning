package rl.env

import cats.effect.{IO, Ref}
import rl.logging.BaseLogger
import scala.annotation.tailrec
import scala.util.Random

/** Blackjack environment
  * Action space: 0 = stick, 1 = hit
  * Observation space: (playerSum, dealerVisibleCard, usableAce) where usableAce is 1 if an ace
  * is being counted as 11 for the player, otherwise 0.
  * Dealer policy: hit until sum >= 17
  * Cards range from (1-10) where 1 is Ace, 10 is face cards. Cards are drawn with replacement
  * - Rewards: +1 player win, 0 draw, -1 player loss
  *
  * Adapted from https://gymnasium.farama.org/environments/toy_text/blackjack/
  */

class BlackJack(
    val stateRef: Ref[IO, (Int, Int, Boolean)],
    logger: BaseLogger[IO]
) extends Env[IO] {
  override type Action = Int
  override type State = (Int, Int, Boolean)

  private def drawCard(): Int = Random.nextInt(10) + 1

  // Dealer policy: hit until sum >= 17
  private def playDealer(dealerVisible: Int): (Int, Boolean) = {
    @tailrec
    def loop(dealerSum: Int, dealerUsableAce: Boolean): (Int, Boolean) = {
      if (dealerSum >= 17) (dealerSum, dealerUsableAce)
      else {
        val newCard = drawCard()
        val (newSum, newUsableAce) =
          BlackJack.addCard(dealerSum, dealerUsableAce, newCard)

        loop(newSum, newUsableAce)
      }
    }

    val dealerInitialUsableAce = dealerVisible == 1
    loop(dealerVisible, dealerInitialUsableAce)
  }

  def reset(): IO[BlackJack] = {
    val (playerSum, dealerVisible, usableAce) = BlackJack.initialise()
    for {
      _ <- stateRef.set((playerSum, dealerVisible, usableAce))
      _ <- logger.debug(
        s"Environment reset. PlayerSum: $playerSum, DealerVisible: $dealerVisible, UsableAce: $usableAce"
      )
    } yield this
  }

  private def comparePlayerDealer(
      playerSum: Int,
      dealerSum: Int
  ): Double = {
    if (playerSum > 21) -1.0
    else if (dealerSum > 21) 1.0
    else if (playerSum > dealerSum) 1.0
    else if (playerSum == dealerSum) 0.0
    else -1.0
  }

  def step(action: Action): IO[(State, Double, Boolean)] = for {
    current <- stateRef.get
    playerSum0 = current._1
    dealerVisible = current._2
    usableAce0 = current._3
    result <-
      if (action == 1) {
        // hit
        val card = drawCard()
        val (newPlayerSum, newUsable) =
          BlackJack.addCard(playerSum0, usableAce0, card)
        val done = newPlayerSum > 21
        val reward = if (done) -1.0 else 0.0
        val newState = (newPlayerSum, dealerVisible, newUsable)
        for {
          _ <- logger.debug(
            s"Agent took action: $action, PlayerSum: $newPlayerSum"
          )
          _ <- stateRef.set(newState)
        } yield (newState, reward, done)
      } else {
        // player stick and dealer's turn
        val (dealerSum, _) = playDealer(dealerVisible)
        val reward = comparePlayerDealer(playerSum0, dealerSum)
        val done = true
        val finalState = (playerSum0, dealerVisible, usableAce0)
        for {
          _ <- logger.debug(
            s"Agent took action: $action, PlayerSum: $playerSum0, DealerSum: $dealerSum, reward: $reward, done: $done"
          )
          _ <- stateRef.set(finalState)
        } yield (finalState, reward, done)
      }
  } yield result

  def getActionSpace: IO[List[Action]] = IO.pure(List(0, 1))

  def getState: IO[State] = stateRef.get
}

object BlackJack {
  private def addCard(
      cardSum: Int,
      usableAce: Boolean,
      drawnCard: Int
  ): (Int, Boolean) = {
    // Step 1: Add the card
    val (sumAfterAdd, aceAfterAdd) =
      if (drawnCard == 1) {
        // Ace
        if (cardSum + 11 <= 21) {
          (cardSum + 11, true)
        } else {
          (cardSum + 1, usableAce)
        }
      } else {
        (cardSum + drawnCard, usableAce)
      }

    // Step 2: Fix bust using usable ace
    if (sumAfterAdd > 21 && aceAfterAdd) {
      (sumAfterAdd - 10, false)
    } else {
      (sumAfterAdd, aceAfterAdd)
    }
  }

  def initialise(): (Int, Int, Boolean) = {
    val card1 = Random.nextInt(10) + 1
    val card2 = Random.nextInt(10) + 1
    val dealerVisible = Random.nextInt(10) + 1

    val initialUsableAce = card1 == 1
    val (playerSum, usableAce) = addCard(card1, initialUsableAce, card2)
    (playerSum, dealerVisible, usableAce)
  }

  def apply(logger: BaseLogger[IO]): IO[BlackJack] = {
    val (playerSum, dealerVisible, usableAce) = initialise()
    for {
      ref <- Ref[IO].of(playerSum, dealerVisible, usableAce)
    } yield new BlackJack(ref, logger)
  }
}
