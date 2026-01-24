ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.17"

val CatsEffectVersion = "3.3.14"
val Http4sVersion = "0.23.16"
val CirceVersion = "0.14.1"
val LogbackVersion = "1.4.14"
val ScalaTestVersion = "3.2.17"
val CatsEffectTestingVersion = "1.5.0"

lazy val root = (project in file("."))
  .settings(
    name := "scala-reinforcement-learning",
    Compile / run / fork := true,
    libraryDependencies ++= Seq(
      "org.typelevel" %% "cats-effect" % CatsEffectVersion,
      "org.http4s" %% "http4s-blaze-server" % Http4sVersion,
      "org.http4s" %% "http4s-circe" % Http4sVersion,
      "org.http4s" %% "http4s-dsl" % Http4sVersion,
      "io.circe" %% "circe-core" % CirceVersion,
      "io.circe" %% "circe-generic" % CirceVersion,
      "io.circe" %% "circe-parser" % CirceVersion,
      "ch.qos.logback" % "logback-classic" % LogbackVersion,
      "org.scalatest" %% "scalatest" % ScalaTestVersion % Test,
      "org.typelevel" %% "cats-effect-testing-scalatest" % CatsEffectTestingVersion % Test
    )
  )
