package rl.agent

import cats.effect.IO
import rl.env.Env
import rl.logging.BaseLogger
import rl.metrics.{EpisodeMetrics, TrainingMetrics}

trait Agent[E <: Env[IO]] {
  protected def logger: BaseLogger[IO]
  def env: E
  def act(state: E#State): IO[E#Action]
  protected def runStep(
      state: E#State,
      action: E#Action
  ): IO[(Boolean, E#State, Double)]

  def runEpisode(): IO[(Double, Int)] = {
    def loop(
        state: E#State,
        reward: Double,
        stepCount: Int
    ): IO[(Double, Int)] =
      for {
        action <- act(state)
        result <- runStep(state, action)
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
      episodeResult <- loop(
        initialState,
        initialReward,
        initialStepCount
      )
    } yield episodeResult
  }

  def learn(episodes: Int): IO[TrainingMetrics] = {
    def loop(
        episode: Int,
        metrics: List[EpisodeMetrics]
    ): IO[List[EpisodeMetrics]] = {
      if (episode >= episodes) IO.pure(metrics)
      else
        for {
          episodeResult <- runEpisode()
          totalEpisodeReward = episodeResult._1
          totalStepCount = episodeResult._2
          episodeMetric = EpisodeMetrics(
            episode + 1,
            totalEpisodeReward,
            totalStepCount
          )
          _ <- logger.info(
            s"Completed episode: ${episode + 1}, Total Reward: $totalEpisodeReward, Total Steps: $totalStepCount"
          )
          updatedMetrics <- loop(episode + 1, metrics :+ episodeMetric)
        } yield updatedMetrics
    }

    loop(0, List.empty).map(metrics => TrainingMetrics(metrics))
  }
}
