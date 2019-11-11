import gym
import numpy as np
from tensorflow import keras
import random

env = gym.make('CartPole-v1')
env.reset()

def model_preparation():
  training_data = []
  accepted_scores = []
  scores = []

  # Getting 1000 games worth of training data
  for game in range(1000):
    # Initalizing variables that each game needs
    score = 0
    game_memory = []
    previous_observation = []
    for step in range(200):      
      # Making a random action (either left or right)
      action = random.randint(0,1)
      # Getting data from the step that has executed the action
      observation, reward, done, info = env.step(action)
      if len(previous_observation) > 0:
        game_memory.append([previous_observation, action])
      previous_observation = observation  
      score += reward # Adding the reward we have gained from this action to our score
      # Ending the game if we have finished
      if done:
        break

    if score >= 50:
      accepted_scores.append(score)
      for data in game_memory:
        if data[1] == 1:
          output = [0,1]
        elif data[1] == 0:
          output = [1,0]        
        training_data.append([data[0],output])

    env.reset()
    scores.append(score)

  #training_data_save = np.array(training_data)
  #np.save('saved.npy', training_data_save)

  return training_data

def neural_network_model(input_size,output_size):
  model = keras.Sequential()
  model.add(keras.layers.Dense(128,input_dim=input_size,activation='relu'))
  model.add(keras.layers.Dense(64,activation='relu'))
  model.add(keras.layers.Dense(output_size,activation='softmax'))
  model.compile(loss='mse',optimizer='Adam')

  return model

def train_model(training_data):
  X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
  y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
  model = neural_network_model(input_size=len(X[0]), output_size=len(y[0]))
  model.fit(X,y,epochs=10)
  return model

trained_model = train_model(model_preparation())
scores = []

env.reset()
for game in range(100):
    # Initalizing variables that each game needs
    score = 0
    previous_observation = []
    for step in range(200):      
      env.render()
      # Making a random action (either left or right)
      if len(previous_observation) > 0:
        action = np.argmax(trained_model.predict(previous_observation.reshape(-1, len(previous_observation)))[0])
      else:        
        action = random.randint(0,1)
      # Getting data from the step that has executed the action
      observation, reward, done, info = env.step(action)
      previous_observation = observation  
      score += reward # Adding the reward we have gained from this action to our score
      # Ending the game if we have finished
      if done:
        scores.append(score)
        break

    env.reset()
    scores.append(score)  

print(scores)













#trained_model = train_model(model_preparation())
#train_model(model_preparation(),False)

