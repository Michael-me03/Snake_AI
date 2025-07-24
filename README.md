# Snake_AI

A Reinforcement Learning project that teaches an AI to play the classic Snake game using a Deep Q-Network (DQN). The goal is for the agent to learn how to survive and collect as much food as possible.

## Project Overview

Algorithm: Deep Q-Learning (DQN)
Language: Python

Libraries: PyTorch, Pygame

Environment: Custom Snake game using Pygame

## Neural Network Architecture

Model: LinearQNet (defined in model.py)
Structure:

Input Layer: 11 features

Hidden Layer: 256 neurons (fully connected, ReLU activation)

Output Layer: 3 values (Q-values for actions)

## Input Features (11 values)
The input state is encoded as a list of 11 boolean/numerical values:

Danger straight ahead

Danger to the right

Danger to the left

Moving left

Moving right

Moving up

Moving down

Food is left of the snake head

Food is right of the snake head

Food is above the snake head

Food is below the snake head

## Output Actions (3 values)

The model predicts Q-values for each possible move:

[1, 0, 0] → move straight

[0, 1, 0] → turn right

[0, 0, 1] → turn left
