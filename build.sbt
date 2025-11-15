ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.17"

val CatsEffectVersion = "3.3.14"

lazy val root = (project in file("."))
  .settings(
    name := "scala-reinforcement-learning",
    libraryDependencies ++= Seq(
      "org.typelevel"   %% "cats-effect"         % CatsEffectVersion,
  )
  )
