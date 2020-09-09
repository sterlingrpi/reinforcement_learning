Reinforcement learning (RL) is method where an agent takes actions to interact with an environment in order to reach a state of high reward. 
Machine learning has been applied to RL as value function (DQN) where the NN predicts the values for each action given the current state. 
This area of machine learning is different than other more traditional types of machine learning (e.g. supervised learning)
because rather than a human telling the NN what the correct action is with labeled data the agent learns the value function by directly
interacting with the environment. RL can be applied to a wide range of problems; strategy based games (i.e. DeedMind’s AlphaGo and OpenAI’s Five),
teaching, healthcare, robotics, autonomous driving, content recommendations, and it is believed my many researchers that RL will eventually 
lead to artificial general intelligence.

This repo contains some of my RL projects. IOne of those being squirrel_hunter, which is a simulated 3D environment using pygame
library and looks similar to the game Doom. The goal of the agent to is keep the squirrel away from the bird feeder. When the squirrel is 
seen a reward of -1 is given. If the bird feeder is seen without the squirrel a reward of +1 is given. An example is given in the youtube
video below. The upper right is the reward for the current state. The lower left is the output from a image segmentation model. The image
segmentation model is used to determine the reward and is also used as a feature extractor for the RL agent model.

[![](http://img.youtube.com/vi/JT37ikDX7xc/0.jpg)](http://www.youtube.com/watch?v=JT37ikDX7xc "")



<img src="https://github.com/sterlingrpi/reinforcement_learning/blob/master/RL_flowchart.jpg" width="500">
