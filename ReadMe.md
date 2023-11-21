## Car Racing DQL Agent

This project contains a Deep Q-Learning (DQL) agent designed to operate in the Car Racing environment. The agent has been trained and tested within the CarRacing environment of the OpenAI Gym library.

# Installation
Clone the project:

```bash
git clone https://github.com/IrohAmca/DQL_Car_Racing.git
cd DQL_Car_Racing
Install the required dependencies:
```
```bash

pip install -r requirements.txt
Train or run the agent:
```
```bash
python RL_Car_Racing.py
```
Usage
RL_Car_Racing.py: Main file responsible for training or running the DQL agent.
File Structure
RL_Car_Racing.py: Main file responsible for training or running the DQL agent.
Car_Racing_Agent.py: File containing the class of the DQL agent.
requirements.txt: List of dependencies.

Parameters
gamma: Discount factor.
learning_rate: Learning rate.
epsilon_decay: Epsilon decay rate.
epsilon_min: Minimum epsilon value.
maxlen: Maximum length for remembered past observations in memory.

Contribution
Fork this repository.
Create a new branch: 
```bash
git checkout -b new-branch.
```
Make your changes and commit them:
```bash
git commit -m 'Add new feature'.
```
Merge your branch with the main repository: git push origin new-branch.
