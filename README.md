
# Snake game

Continuation of my master's degree project. I added gui which helps user to start snake game or helps create Q-learning / deep Q-learning agent.
## Acknowledgements

 - [Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0&ab_channel=ArxivInsights)
 - [Q-Learning on YT](https://www.youtube.com/watch?v=qhRNvCVVJaA&ab_channel=deeplizard)
 - [Deep Q-Learning on YT](https://www.youtube.com/watch?v=wrBUkpiRvCA&t=540s&ab_channel=deeplizard)

## Setup

You can create new virtual environment in folder with all project files by typing in terminal

```
python -m venv venv
```

and activate it using command:

- for Windows

```
venv\Scripts\actibate
```

- for other OSes

```
source venv/bin/activate
```

To run the project you need to install required packages, which are included in requirements.txt file

```
pip install -r requirements.txt
```
## Launch

To launch the project type command

```
python gui.py
```
## Controls

At the beginning select 1 of 3 options in top left part of window. Then fill all filds that pops out and press START button. 

If you selected 'Play game' you will control snake with arrows. 

## Screenshots
Main window 
![](images/main_window.png)
Play game view
![](images/player_game.png)
Create Q-learning window view
![](images/ql_game.png)
Create deep Q-learning window view
![](images/dql_game.png)
Snake game view
![](images/gameplay.png)

