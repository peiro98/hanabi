import argparse
import logging
import time
import sys
import os

from implementation.learning import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Double Deep Q-Learning network to play Hanabi.",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--players",
        action="store",
        dest="n_players",
        type=int,
        choices=[2, 3, 4, 5],
        default=2,
        required=True,
        help="Number of players",
    )
    parser.add_argument(
        "--training-players",
        action="store",
        dest="n_training_players",
        type=int,
        default=5,
        help="Number of training players",
    )
    parser.add_argument(
        "--iterations",
        action="store",
        dest="n_iterations",
        type=int,
        default=500_000,
        help="Number of iterations to run. Similar projects suggest to run at least tens of millions of episodes",
    )
    parser.add_argument(
        "--discount",
        action="store",
        dest="discount",
        default=0.25,
        type=float,
        help="Discount factor for the Q value of the next state",
    )
    parser.add_argument(
        "--batch-size",
        action="store",
        dest="batch_size",
        default=64,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--initial-eps",
        action="store",
        dest="initial_eps",
        default=1.0,
        type=float,
        help="Initial epsilon coefficient (probability of selecting a random action)",
    )
    parser.add_argument(
        "--eps-step",
        action="store",
        dest="eps_step",
        default=0.99995,
        type=float,
        help="Epsilon coefficient decay step",
    )
    parser.add_argument(
        "--minimum-step",
        action="store",
        dest="minimum_eps",
        default=0.1,
        type=float,
        help="Minimum value of the epsilon coefficient",
    )
    parser.add_argument(
        "--turn-dependent-eps",
        action="store_true",
        dest="turn_dependent_eps",
        help="If present, the epsilon coefficient is a function of the turn index",
    )
    parser.add_argument(
        "--target-refresh-interval",
        action="store",
        dest="target_model_refresh_interval",
        type=int,
        default=1000,
        help="Interval between the updates of the target model",
    )
    parser.add_argument(
        "--evaluation-interval",
        action="store",
        dest="evaluation_interval",
        type=int,
        default=250,
        help="Interval between evaluations of the model",
    )
    parser.add_argument(
        "--evaluation-num-games",
        action="store",
        dest="evaluation_num_games",
        type=int,
        default=100,
        help="Number of games to simulate during the evaluation phase",
    )
    parser.add_argument(
        "--initial-lr", action="store", dest="initial_lr", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument(
        "--lr-step", action="store", dest="lr_step", type=int, default=25_000, help="Learning rate step"
    )
    parser.add_argument(
        "--lr-gamma", action="store", dest="lr_gamma", type=float, default=0.5, help="Learning rate gamma factor"
    )
    parser.add_argument(
        "--model-save-path", action="store", dest="model_save_path", type=str, help="Destination path for the model"
    )

    # Parse the arguments
    args = vars(parser.parse_args())

    # Setup logging
    if not os.path.exists("logs"):
        os.mkdir("logs")
    filename = time.strftime("%Y_%m_%d-%I_%M_%S_%p")
    logging.basicConfig(
        filename=f"logs/{filename}.log",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("Starting a new training process with args: {")
    for key, value in args.items():
        logging.info(f"  {key}: {value}")
    logging.info("}")

    # and start training
    train(args)
