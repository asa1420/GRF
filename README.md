# Google Research Football
This is my Final Year Project as a Bachelor of Engineering (Electronic & Electrical Engineering) Student at University College London.

In this project, a game environment called Google Research Football was chosen to research the possibility of
cooperation between agents using multi-agent Reinforcement Learning. A series of experiments were conducted
that led to step-by-step improvements in the Reinforcement Learning models. Starting with single-agent
experiments on empty goal scenarios in the game, up to multi-agent experiments in 3 vs 1 scenarios similar to those
in the literature. The main findings from the project include that when using Tensorflow and Keras, predictions made
using Model() are more efficient (35x faster) than using Model.predict(). Another main result is that the checkpoint
reward function is a disadvantage in some certain cases, when the scenario is too easy, since it leads to exploitation
and overfitting. As for model training accomplishments, in the empty goal scenario, the issue of overfitting caused
the scenario to be solved within 64k steps (compared to 1M steps in the literature), as the player only performs one
action which is running straight into the goal. As for the main multi-agent result, a 2-player model learnt to pass
through a defender by passing to each other, after training for 5M steps.

This repository will be of great use to anyone interested is starting off with Reinforcement Learning research using Google Research Football, as I include every obstacle I went throught in my lab book, which is something I find really valuable for any beginner in this field of research.


Feel free to use any piece of code I produced, or experimental setups created! Kindly reference my repository or final report once doing so.
