# Scala Reinforcement Learning API

A REST API for training reinforcement learning agents on various environments.

## Running the Server

Start the server with:

```bash
sbt run
```

The server will start on `http://localhost:8080`


## Train Agent

Train a reinforcement learning agent by sending a POST request to `/train` with JSON configuration.

**Endpoint:** `POST /train`

### Example Request

```bash
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{
    "environment": {
      "type": "BlackJack"
    },
    "agent": {
      "type": "Sarsa",
      "learningRate": 0.1,
      "discountFactor": 0.9
    },
    "exploration": {
      "type": "UCB",
      "constant": 1
    },
    "episodes": 100
  }'
```


## Environments

### GridWorld1D
Simple 1D grid world with 5 positions.
```json
{
  "type": "GridWorld1D"
}
```

### BlackJack
Classic Blackjack game.
```json
{
  "type": "BlackJack"
}
```

### FrozenLake
4x4 grid with holes and slippery surfaces.
```json
{
  "type": "FrozenLake",
  "isSlippery": true,
  "successRate": 0.7
}
```

## Agents

### Q-Learning
```json
{
  "type": "QLearning",
  "learningRate": 0.1,
  "discountFactor": 0.9
}
```

### Sarsa
```json
{
  "type": "Sarsa",
  "learningRate": 0.1,
  "discountFactor": 0.9
}
```

## Exploration Methods

### Epsilon-Greedy
```json
{
  "type": "EpsilonGreedy",
  "explorationRate": 0.1
}
```

### UCB (Upper Confidence Bound)
```json
{
  "type": "UCB",
  "constant": 1
}
```