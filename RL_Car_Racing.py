import gymnasium as gym
import numpy as np
from Car_Racing_Agent import DQLAgent
import cv2

env = gym.make("CarRacing-v2", domain_randomize=True,render_mode="human")

reward_t = 0
score = 0
rewards = []

reward=0
if __name__ == "__main__":
    agent = DQLAgent(env, gamma=0.95, learning_rate=0.1, epsilon_decay=0.995, epsilon_min=0.01, maxlen=10000,loaded=True,path="C:/Users/aliay/RL_çalisma/F1_20_11_23.keras")

    batch_size = 65
    episodes = 50
    for e in range(episodes):
        step=0
        state, info = env.reset(options={"randomize": False},seed=1)
        env.render()
        reward_temp=0
        
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state=state/128
        while True:
            step+=1
            # act
            action = agent.act(state)  # select an action
            # Ensure that action is an array
            if not isinstance(action, np.ndarray):
                action = np.array([action])

            # step
            """
            action[1] = np.clip(action[1] + 0.3, -1.0, 1.0)
            action[2] = np.clip(action[2] - 0.3, -1.0, 1.0)
            
            if action[1]>0.6:
                reward+=5"""
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # RGB to Gray
           
            next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            next_state=next_state/128
            
            
            # Save memory
            agent.remember(state, action, reward, next_state, truncated, terminated)
            
            # Update state
            state=next_state
            
            # Update the epsilon
            agent.adaptiveEGreedy()
            
            # Save rewards
            reward_t += reward
            reward_temp += reward
            rewards.append(reward)
            if step==30:
                agent.replay(batch_size)
            if terminated or truncated or reward_temp<-40:
                agent.replay(batch_size)
                state, info = env.reset(options={"randomize": False},seed=1)
                print("Episode: {}, reward: {},  info: {}".format(e, reward_t, info))

                

                reward_t = 0
                break

    env.close()
    
agent.model.save("F1_22_11_23.keras")
