# Asynchronous Methods for Deep Reinforcement Learning in TensorFlow + OpenAI Gym
This is an implementation of Asynchronous Methods for Deep Reinforcement Learning (based on [Mnih et al., 2016](https://arxiv.org/abs/1602.01783)) in TensorFlow + OpenAI Gym.  

## Requirements
- gym (Atari environment)
- scikit-image
- tensorflow

## Results
Coming soon...

## Usage
#### Training

* It is suggested to run the code in a new environment.

After entering the new environment, in the terminal, run:
```
pip install -r requirements.txt
```


For asynchronous advantage actor-critic, run:

```
python main.py
```

#### Visualizing learning with TensorBoard
Run the following:

```
tensorboard --logdir=summary/
```

## References
- [Mnih et al., 2016, Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

## Team Members
* Aditya Suryawanshi - 210150004
* Aryan Singh - 210150008
* Palthiya Laanith Chouhan - 210150014
