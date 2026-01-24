# Scala Reinforcement Learning

Inspired by [Farama Gymnasium](https://gymnasium.farama.org/), this library provides various environments for training reinforcement learning agents. Unlike Gymnasium, the algorithms are built-in, allowing users to easily plug-and-play different approaches and compare their performance across environments through a simple API.

## Running the Server

Start the server with:

```bash
sbt run
```

The server will start on `http://localhost:8080`


## Train Agent

Train a reinforcement learning agent by sending a POST request to `/train` with JSON configuration. There are 4 settings that a user has to configure:

1. **Environment**: The RL environment where the agent will learn (e.g., GridWorld1D, BlackJack)
2. **Agent**: The reinforcement learning algorithm to use (e.g., Q-Learning, Sarsa)
3. **Exploration**: The exploration strategy that balances exploration vs. exploitation (e.g., Epsilon-Greedy, UCB)
4. **Episodes**: The number of training episodes to run

Full list of configurations for each setting is specified in the subsequent sections.

The response contains:
- **status**: "success" if training completed successfully
- **error**: null on success, contains error message on failure
- **metrics**:
  - **episodeMetrics**: List of metrics for each episode
    - **episodeNumber**: Episode number
    - **totalReward**: Total reward accumulated in the episode
    - **totalSteps**: Number of steps taken in the episode


### Example Request

```bash
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{
    "environment": {
      "type": "GridWorld1D"
    },
    "agent": {
      "type": "Sarsa",
      "learningRate": 0.1,
      "discountFactor": 0.9
    },
    "exploration": {
      "type": "EpsilonGreedy",
      "explorationRate": 0.1
    },
    "episodes": 100
  }'
```

### Example Response

#### Success Response

```json
{
  "status": "success",
  "error": null,
  "metrics": {
    "episodeMetrics": [
      {
        "episodeNumber": 1,
        "totalReward": 5.0,
        "totalSteps": 10
      },
      {
        "episodeNumber": 2,
        "totalReward": 8.0,
        "totalSteps": 8
      }
    ]
  }
}
```

#### Error Response

```json
{
  "status": "error",
  "error": "Error message describing what went wrong",
  "metrics": null
}
```

## Environments

### GridWorld1D
Simple 1D grid world with 5 positions where an agent either move left or right. Episode terminate when agent reaches extreme right starting from the middle
```json
{
  "type": "GridWorld1D"
}
```

### BlackJack
Classic Blackjack game where agent must beat the dealer by obtaining cards that sum closer to 21 as compared to dealer
```json
{
  "type": "BlackJack"
}
```

### FrozenLake
4x4 grid with holes and slippery surfaces. Agent must successfully navigate to the goal position by moving either up/down/left/right while avoiding the holes. 
If `isSlippery` is True, then agent might not move in the same direction as what it intended to do. Probability of moving in the intended direction is specified by 
`successRate` with (1-successRate) being the probability of moving in any of the other directions.
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
  "discountFactor": 0.9,
  "nSteps" : 1
}
```

### Sarsa
```json
{
  "type": "Sarsa",
  "learningRate": 0.1,
  "discountFactor": 0.9,
  "nSteps" : 1
}
```

### Double Q-Learning
```json
{
  "type": "DoubleQLearning",
  "learningRate": 0.1,
  "discountFactor": 0.9,
  "nSteps" : 1
}
```

### Dyna-Q
```json
{
  "type": "DynaQ",
  "learningRate": 0.1,
  "discountFactor": 0.9,
  "planningSteps" : 3
}
```

### Dyna-Q+
```json
{
  "type": "DynaQPlus",
  "learningRate": 0.1,
  "discountFactor": 0.9,
  "planningSteps" : 3
  "kappa" : 0.001,
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