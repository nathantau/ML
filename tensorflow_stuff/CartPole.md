# CartPole Environment (OpenAI-Gym)

## Creating the environment

Let's start by importing the gym package.

```Python
pip install gym
...
import gym
```

Now let's create the actual environment. We will also have to reset the environment. 

```Python
env = gym.make('CartPole-v1')
env.reset()
```

## Playing in the environment

We will then iterate through multiple 'plays' in the environment using random steps and actions. We want the max number of steps to be 200.

```Python
for step in range(200):
  
  # Renders the environment for us to see
  env.render()

  # Picks a random action for us to use
  action = env.action_space.sample()

  # Returns data and performs an action
  observation, reward, done, info = env.step(action)

  # Printing out important information gained after our last action
  print("Step {}:".format(step))
  print("action: {}".format(action))
  print("observation: {}".format(observation))
  print("reward: {}".format(reward))
  print("done: {}".format(done))
  print("info: {}".format(info))

  # If we 'die', we exit this play
  if done:
    break
```

We will get important information back:
*action: This will be either a 0 or a 1, depending on if we move left or right
*observation: This will be a list of the cart position, cart velocity, pole angle, and pole velocity at the tip
*reward: The value of the reward we are gaining from successfully staying alive after our action
*done: True or False, whether or not we have 'died' and the game is 'over' or 'done'
*info: Relevant information that is returned

Let's note that the game is over when:
*the pole angle is + or - 12 degrees
*the cart position is + or - 2.4, meaning that it is 2.4 from the center
*the episode length is greater than 200

## Setting up our model

We will need to import a few more packages for our model.

```Python
import numpy as np
from tensorflow import keras
```

