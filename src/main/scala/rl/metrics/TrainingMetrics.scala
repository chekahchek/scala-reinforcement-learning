package rl.metrics

case class EpisodeMetrics(
    episodeNumber: Int,
    totalReward: Double,
    totalSteps: Int
)

case class TrainingMetrics(
    episodeMetrics: List[EpisodeMetrics]
)
