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
         
        """if self.model_load==False:
            self.model = self.build_model()
        if self.model_load==True:
            self.model=self.load_model(path)"""
        self.model_d,self.model_f=self.build_model()
        
            
        
    def build_model(self):
        # neural network for deep q learning
        model_f = Sequential()
        model_f.add(Conv2D(32, (8, 8), strides=(3,3),activation='relu', input_shape=(96,96,1)))
        model_f.add(Conv2D(64, (4, 4),strides=(3,3), activation='relu'))
        model_f.add(Conv2D(128, (3, 3),strides=(3,3), activation='relu'))
        model_f.add(Flatten())
        #model.add(Dense(512, activation='relu'))
        model_f.add(Dense(128, activation='relu'))
        model_f.add(Dropout(0.2))
        model_f.add(Dense(2, activation='linear'))  # Q-değerleri için lineer aktivasyon
        model_f.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=self.learning_rate))
        
        model_d=Sequential()
        model_d.add(Conv2D(32, kernel_size=(8,8), strides=(3,3),activation='relu',input_shape=(96,96,1)))
        model_d.add(Conv2D(64,kernel_size=(8,8),strides=(3,3),activation='relu'))
        model_d.add(Conv2D(128, (3, 3),strides=(3,3), activation='relu'))
        model_d.add(Flatten())
        model_d.add(Dense(128, activation='relu'))
        model_d.add(Dropout(0.2))
        model_d.add(Dense(1, activation='linear'))
        model_d.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model_d,model_f


    def remember(self, state, action, reward, next_state, truncated,terminated):
        # storage
        self.memory.append((state, action, reward, next_state, truncated,terminated))

    def act(self, state):
        # acting: explore or exploit
        if random.uniform(0, 1) <= self.epsilon:
            # Rasgele bir aksiyon oluştur
            return np.random.uniform(-1, 1, size=(3,))
        else:
            act_value_d = self.model_d.predict(np.expand_dims(state, axis=0))
            act_value_f = self.model_f.predict(np.expand_dims(state, axis=0))
            act_values=act_value_d+act_value_f
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
    
        minibatch = random.sample(self.memory, batch_size)
    
        for state, action, reward, next_state, truncated, terminated in minibatch:
            if truncated:
                target_d = reward
                target_f = reward
            else:
                target_d = reward + self.gamma * np.amax(self.model_d.predict(np.expand_dims(next_state, axis=0))[0])
                target_f = reward + self.gamma * np.amax(self.model_f.predict(np.expand_dims(next_state, axis=0))[0])
            train_target_d = self.model_d.predict(np.expand_dims(state, axis=0), verbose=0)
            train_target_f= self.model_f.predict(np.expand_dims(state, axis=0), verbose=0)
            train_target_d[0] = target_d
            train_target_f[0] = target_f
            if reward>-20:
                # Use actual training data and labels for fitting the model
                self.model_d.train_on_batch(np.expand_dims(state, axis=0), train_target_d)
                self.model_f.train_on_batch(np.expand_dims(state, axis=0), train_target_f)

    def load_model(self,path):   
        model=keras.models.load_model(path)
        self.model=model
        return model
        
    def adaptiveEGreedy(self):
     # Adjust epsilon based on the decay rate until it reaches epsilon_min
     if self.epsilon > self.epsilon_min:
         self.epsilon *= self.epsilon_min + (0.99 - self.epsilon_min) * math.exp(-math.log(0.001))

            
    
        
        