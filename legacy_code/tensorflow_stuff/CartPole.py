import gym
import numpy as np
from tensorflow import keras
import random

def get_training_data(number_games, number_steps, accepted_score):

  training_data = []

  env = gym.make('CartPole-v1')
  env.reset()

  for game in range(number_games):
    score = 0
    memory = []
    prev_obs = []

    for steps in range(number_steps):
      action = random.randint(0, 1)
      obs, reward, done, info = env.step(action)

      if len(prev_obs) > 0:
        memory.append([prev_obs, action])
      
      prev_obs = obs
      score += reward

      if done:
        break

    if score > accepted_score:
      for data in memory:
        if data[1] == 0:
          output = [1, 0]
        else:
          output = [0, 1]

        training_data.append([data[0], output])
    
    env.env.close()
    env.reset()

  env.close()
  return training_data  

def create_neural_network(input_size, output_size):

  num_neurons = (input_size + output_size) / 2

  model = keras.Sequential()
  model.add(keras.layers.Dense(num_neurons, input_dim=input_size, activation='relu'))
  model.add(keras.layers.Dense(output_size, activation='softmax'))
  model.compile(loss='mse', optimizer='Adam')

  return model

def train_neural_network(model, training_data):

  X = [i[0] for i in training_data]
  X = np.array(X).reshape(-1,4)
  print(X)
  print(X.shape)
  y = [i[1] for i in training_data]
  y = np.array(y).reshape(-1,2)
  print(y)
  print(y.shape)

  model.fit(X, y, epochs=10)
  return model

def play_game(model, num_games, num_steps):

  scores = []

  env = gym.make('CartPole-v1')
  env.reset()

  for game in range(num_games):
    score = 0
    memory = []
    prev_obs = []

    for steps in range(num_steps):      

      if steps == 0:
        action = random.randint(0, 1)
      else:
        action = np.argmax(model.predict(np.array(prev_obs).reshape(-1,4)))
        print(f'Previous Observation: {np.array(prev_obs).reshape(-1,4)}')
        print(action)

      obs, reward, done, info = env.step(action)
      
      prev_obs = obs
      score += reward

      if done:
        scores.append(score)
        break

    env.env.close()
    env.reset()

  env.close()
  print(scores)


model = create_neural_network(4, 2)
training_data = get_training_data(100, 200, 50)
model = train_neural_network(model, training_data)
play_game(model, 100, 200)









