# Computational Intelligence 2021-2022

Exam of computational intelligence 2021 - 2022. It requires teaching the client to play the game of Hanabi (rules can be found [here](https://www.spillehulen.dk/media/102616/hanabi-card-game-rules.pdf)).

## Deep Q-Learning
The Hanabi problem is tackled using a Deep Reinforcement Learning approach inspired by the Deepmind's work on Atari games ([1](https://arxiv.org/pdf/1312.5602v1.pdf) and [2](https://arxiv.org/pdf/1509.06461.pdf)). The proposed solution implements a Double DQN approach where two neural networks are used to respectively select and evaluate the actions performed by the greedy policy. 

The following equation determines the target Q value for the current state and the selected action (S, a). The best action to be performed in the next state S<sub>t+1</sub> is selected according to the online network θ<sub>t</sub> and evaluated on the offline (target) network θ<sub>t</sub><sup>-</sup>. The target network is a copy of the online network at fixed time instances.

<img src="https://render.githubusercontent.com/render/math?math=%7B%5Ccolor%7Bgray%7DY_%7Btarget%7D%20%3D%20R_%7Bt%2B1%7D%20%2B%20%5Cgamma%20Q%20(S_%7Bt%2B1%7D%2C%20%5Cargmax_%7Ba%7D%20Q(S_%7Bt%2B1%7D%2C%20a%3B%20%5Ctheta_t)%2C%20%5Ctheta_t%5E%7B-%7D)%7D">


Furthermore, experience replay is used to keep a memory of the last 384k observed pairs of *(state, action, reward, next_state)*. At the end of each game, the states collected are moved to the experience replay memory. Then, a batch is sampled from the memory and used to train the network.

>DISCLAIMER: the performance of the DQN agent using the pre-trained models under the `rl-models` are *really* poor mainly due to the limited time available for training. Deepmind's work shows that RL agents requires very long training times (~200M frames for each Atari game, one week worth of computation on a GPU) with huge experience replay memories. I was very limited by the available computational power and trained the models for only 50k games. However, the loss (computed as the mean squared error between the outputs of the online network and the target values) shows a decreasing trend as expected. Indeed, I believe this repository represents a strong baseline for possible future work on the topic. Training logs can be viewed [here](https://drive.google.com/drive/folders/19sTkb_kl4vd4Ipx-TuEJHixqoJte2vxW?usp=sharing).

### Possible future improvements
Some of the possible future improvements over the baseline provided at the moment include:
- Different encodings for the game state as neural network input (i.e. 2d/3d structures), possibly taking into account not only the current state but also the previous n.
- Different network architectures: RNNs could be useful to adapt the Q output of the network according to previous actions performed by the other player resulting in an adaptive approach.
- More complex sampling strategies for the experience replay memory, possibly with data augmentation on the observed states.

**NOTE**: training can be started with the `train_all.sh` script. PyTorch is required.

## Server

The server accepts passing objects provided in GameData.py back and forth to the clients.
Each object has a ```serialize()``` and a ```deserialize(data: str)``` method that must be used to pass the data between server and client.

Watch out! I'd suggest to keep everything in the same folder, since serialization looks dependent on the import path (thanks Paolo Rabino for letting me know).

Server closes when no client is connected.

To start the server:

```bash
python server.py <minNumPlayers>
```

Arguments:

+ minNumPlayers, __optional__: game does not start until a minimum number of player has been reached. Default = 2


Commands for server:

+ exit: exit from the server

## Client

To start the server:

```bash
python client.py <IP> <port> <PlayerName>
```

Arguments:

+ IP: IP address of the server (for localhost: 127.0.0.1)
+ port: server TCP port (default: 1024)
+ PlayerName: the name of the player

Commands for client:

+ exit: exit from the game
+ ready: set your status to ready (lobby only)
+ show: show cards
+ hint \<type> \<destinatary>:
  + type: 'color' or 'value'
  + destinatary: name of the person you want to ask the hint to
+ discard \<num>: discard the card *num* (\[0-4]) from your hand
