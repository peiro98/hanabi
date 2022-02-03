#!/usr/bin/env python3

# Reinforcement Learning based agent.
# The implementation and training of the agent can be found
# under the core/ directory. The client communicates with the 
# agent performing the decisions through a Proxy class that is
# implemented in client_proxy.py
#
# Simone Alberto Peirone (S286886)

from sys import argv
import logging
import sys
import time

import socket

import GameData
from constants import *

from core.cards import CARD_COLORS, CARD_COLORS_EXTENDED, Card
from core.moves import DiscardMove, HintColorMove, HintValueMove, PlayMove
from client_proxy import Proxy
from core.players import DRLNonTrainableAgent

if len(argv) < 4:
    print("You need the player name to start the game.")
    # exit(-1)
    playerName = "Test"  # For debug
    ip = HOST
    port = PORT
else:
    playerName = argv[3]
    ip = argv[1]
    port = int(argv[2])


# Setup logging
logging.basicConfig(
    filename=f"game-client.log",
    level=logging.DEBUG,
    format=f"%(asctime)s %(levelname)s: [{playerName}] %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# Game status
run = True
statuses = ["Lobby", "Game", "GameHint"]
status = statuses[0]


# Proxy to the reinforcement learning agent: a proxy
# is required since agents are defined in a game environment
# independent way. A proxy allows the agent to access the relevant
# informations it needs in order to take its decisions.
player = DRLNonTrainableAgent(
    playerName,
    filenames={
        2: "rl-models/DQN_2_players.npy",
        3: "rl-models/DQN_3_players.npy",
        4: "rl-models/DQN_4_players.npy",
        5: "rl-models/DQN_5_players.npy",
    },
)
proxy = Proxy(player)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # setup a connection with the server
    request = GameData.ClientPlayerAddData(playerName)
    s.connect((HOST, PORT))
    s.send(request.serialize())

    data = s.recv(DATASIZE)

    data = GameData.GameData.deserialize(data)
    if type(data) is GameData.ServerPlayerConnectionOk:
        logging.debug("Connection accepted by the server. Welcome " + playerName)

    # display the prompt (only for interactive players)
    # logging.debug("[" + playerName + " - " + status + "]: ")

    # notify the server that the player is ready to start a new game
    s.send(GameData.ClientPlayerStartRequest(playerName).serialize())

    buffer = []
    while run:
        dataOk = False
        
        # data = s.recv(DATASIZE)
        # if not data:
        #     continue

        # Some issues regarding incomplete states were seen during
        # experiments. I was not able to track down the origin of the
        # issues but implementing the following buffering approach
        # over the data received from the server seems to solve everything.
        # Incoming packets are collected in the buffer and the current
        # players performs actions only when the turn is correct and there 
        # are no pending packets in the buffer, meaning everything that was
        # supposed to be processed was indeed processed.
        while True:
            s.settimeout(0.01)
            try:
                _data = s.recv(DATASIZE)
                # read from the socket with 0.01 sec timeout
                buffer.append(_data)
            except socket.timeout:
                s.settimeout(None)
                break

        if len(buffer) == 0:
            continue

        # pick the oldest entry from the buffer
        data = buffer.pop(0)
        logging.debug(f"Number of pending packets: {len(buffer)}")

        data = GameData.GameData.deserialize(data)
        logging.debug(f"Got packet: {data}")

        if type(data) is GameData.ServerPlayerStartRequestAccepted:
            dataOk = True
            logging.info("Ready: " + str(data.acceptedStartRequests) + "/" + str(data.connectedPlayers) + " players")

            # wait a ServerStartGameData packet
            # data = s.recv(DATASIZE)
            # data = GameData.GameData.deserialize(data)

        if type(data) is GameData.ServerStartGameData:
            dataOk = True
            logging.info("Game start!")

            # refresh the game state
            s.send(GameData.ClientGetGameStateRequest(playerName).serialize())

            # send 'ready' to start
            s.send(GameData.ClientPlayerReadyData(playerName).serialize())
            status = statuses[1]

            # add players to the proxy
            for name in data.players:
                proxy.add_player(name)

        if type(data) is GameData.ServerGameStateData:
            dataOk = True

            # print the current game state (as seen by current player)
            log = "Current player: " + data.currentPlayer + "\n"
            log += "Player hands: \n"
            for p in data.players:
                log += p.toClientString() + "\n"

            log += "Cards in your hand: " + str(data.handSize) + "\n"
            log += "Table cards: \n"
            for pos in data.tableCards:
                log += pos + ": [ \n"
                for c in data.tableCards[pos]:
                    log += c.toClientString() + "\n"
                log += "]\n"

            log += "Discard pile: \n"
            for c in data.discardPile:
                log += "\t" + c.toClientString() + "\n"

            log += "Note tokens used: " + str(data.usedNoteTokens) + "/8\n"
            log += "Storm tokens used: " + str(data.usedStormTokens) + "/3\n"
            logging.debug(log)

            for p in data.players:
                # the hand of the player as it is currently known by the prxy
                _, current_known_hand = proxy.players[p.name]

                if p.name == proxy.player.name:
                    # add unknown card(s)
                    for _ in range(data.handSize - len(current_known_hand)):
                        proxy.append_card_to_player_hand(p.name, None, None)
                else:
                    # add new cards to this player's hand (required after a play or discard move)
                    if len(current_known_hand) < len(p.hand):
                        for i in range(len(current_known_hand), len(p.hand)):
                            proxy.append_card_to_player_hand(p.name, p.hand[i].color, p.hand[i].value)

            # the game was originally developed using the opposite semantic
            # proxy.blue_tokens is the number of available blue tokens
            proxy.blue_tokens = 8 - data.usedNoteTokens
            proxy.red_tokens = 3 - data.usedStormTokens

            if data.currentPlayer == proxy.player.name and len(buffer) == 0:
                # it's my turn
                try:
                    # ask the agent to perform an action
                    action = proxy.step()
                    logging.info(f"Playing {action}")

                    if isinstance(action, PlayMove):
                        s.send(GameData.ClientPlayerPlayCardRequest(playerName, action.index).serialize())
                    elif isinstance(action, HintColorMove):
                        color = CARD_COLORS_EXTENDED[CARD_COLORS.index(action.color)].lower()
                        s.send(GameData.ClientHintData(playerName, action.player, "color", color).serialize())
                    elif isinstance(action, HintValueMove):
                        s.send(GameData.ClientHintData(playerName, action.player, "value", action.value).serialize())
                    elif isinstance(action, DiscardMove):
                        s.send(GameData.ClientPlayerDiscardCardRequest(playerName, action.index).serialize())
                    else:
                        raise TypeError("Player was not able to provide a valid action")
                except:
                    logging.error("Player was not able to select a valid action")
                    # code should never reach this point
                    # play a random card to allow the game to proceed
                    s.send(GameData.ClientPlayerPlayCardRequest(playerName, 0).serialize())

        if type(data) is GameData.ServerActionInvalid:
            dataOk = True
            logging.error(f"Invalid action performed. Reason: {data.message}")

            # refresh the game state
            s.send(GameData.ClientGetGameStateRequest(playerName).serialize())

        if type(data) is GameData.ServerActionValid:
            dataOk = True
            logging.debug("Action valid!")
            logging.debug("Current player: " + data.player)

            if data.action == "discard":
                # move the card to the discard pile
                proxy.discard_pile.append(Card(data.card.color[0].upper(), data.card.value))
                # and remove the card from the player's hand
                _, hand = proxy.players[data.lastPlayer]
                del hand[data.cardHandIndex]

            # refresh the game state
            s.send(GameData.ClientGetGameStateRequest(playerName).serialize())

        if type(data) is GameData.ServerPlayerMoveOk:
            dataOk = True
            logging.debug("Nice move!")
            logging.debug("Current player: " + data.player)

            # update the board state
            color, value = data.card.color[0].upper(), data.card.value
            proxy.append_card_to_board(color, value)
            # and remove the card from the player's hand
            _, hand = proxy.players[data.lastPlayer]
            del hand[data.cardHandIndex]

            # refresh the game state
            s.send(GameData.ClientGetGameStateRequest(playerName).serialize())

        if type(data) is GameData.ServerPlayerThunderStrike:
            dataOk = True
            logging.debug("OH NO! The Gods are unhappy with you!")

            # move the card to the discard pile
            color, value = data.card.color[0].upper(), data.card.value
            proxy.append_card_to_discard_pile(color, value)

            # and remove the card from the player's hand
            _, hand = proxy.players[data.lastPlayer]
            del hand[data.cardHandIndex]

            # refresh the game state
            s.send(GameData.ClientGetGameStateRequest(playerName).serialize())

        if type(data) is GameData.ServerHintData:
            dataOk = True
            logging.debug("Hint type: " + data.type)
            logging.debug(
                "Player "
                + data.destination
                + " cards with value "
                + str(data.value)
                + " are: "
                + ", ".join(map(str, data.positions))
            )

            # assign the hints
            proxy.hint_player(data.destination, data.type, data.value, data.positions)

            # refresh the game state
            s.send(GameData.ClientGetGameStateRequest(playerName).serialize())

        if type(data) is GameData.ServerInvalidDataReceived:
            dataOk = True
            logging.error(data.data)

            # refresh the game state
            s.send(GameData.ClientGetGameStateRequest(playerName).serialize())

        if type(data) is GameData.ServerGameOver:
            dataOk = True

            logging.info(f"Game completed with score {data.score}")

            # clean up the proxy
            proxy.clean()

            # logging.info("Ready for a new game!")
            
            # done
            run = False

            # start again
            # time.sleep(.5)
            # s.send(GameData.ClientPlayerStartRequest(playerName).serialize())

        if not dataOk:
            logging.error("Unknown or unimplemented data type: " + str(type(data)))

        # display the prompt (only for interactive players)
        # print("[" + playerName + " - " + status + "]: ", end="")
