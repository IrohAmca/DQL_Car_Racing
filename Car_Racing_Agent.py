import keras 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout
from keras.optimizers import Adam
from collections import deque
import random
import math


class DQLAgent():
    def __init__(self, env,gamma,learning_rate,epsilon_decay,epsilon_min,maxlen,loaded=False,path=None):
        # parameter / hyperparameter
        #self.state_size = 2
        #self.action_size = env.action_space.n

        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.epsilon = 1  # explore
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.memory = deque(maxlen=maxlen)
        self.model_load=loaded
        
        if self.model_load==False:
            self.model = self.build_model()
        if self.model_load==True:
            self.model=self.load_model(path)
            
        
    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(3,3),activation='relu', input_shape=(96,96,1)))
        model.add(Conv2D(64, (4, 4),strides=(3,3), activation='relu'))
        model.add(Conv2D(128, (3, 3),strides=(3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='linear'))  # Q-değerleri için lineer aktivasyon
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, truncated,terminated):
        # storage
        self.memory.append((state, action, reward, next_state, truncated,terminated))

    def act(self, state):
        # acting: explore or exploit
        if random.uniform(0, 1) <= self.epsilon:
            # Rasgele bir aksiyon oluştur
            return np.random.uniform(-1, 1, size=(3,))
        else:
            act_values = self.model.predict(np.expand_dims(state, axis=0))
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
    
        minibatch = random.sample(self.memory, batch_size)
    
        for state, action, reward, next_state, truncated, terminated in minibatch:
            if truncated:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0))[0])
    
            train_target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            train_target[0] = target
            if reward>-20:
                # Use actual training data and labels for fitting the model
                self.model.train_on_batch(np.expand_dims(state, axis=0), train_target)


    def load_model(self,path):   
        model=keras.models.load_model(path)
        self.model=model
        return model
        
    def adaptiveEGreedy(self):
     # Adjust epsilon based on the decay rate until it reaches epsilon_min
     if self.epsilon > self.epsilon_min:
         self.epsilon *= self.epsilon_min + (0.99 - self.epsilon_min) * math.exp(-math.log(0.001))

            
    
        
        