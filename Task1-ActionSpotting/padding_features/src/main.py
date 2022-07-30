import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClips, SoccerNetClipsTesting  # ,SoccerNetClipsOld
from model import Model
from train import trainer, test, testSpotting
from loss import NLLLoss
from SoccerNet.Downloader import getListGames


def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))
    # load list games from npy
    if(args.train_list is ""):
        train_list = np.array(getListGames(['train']))
    else:
        train_list = np.load(args.train_list)

    if(args.test_list is ""):
        test_list = np.array(getListGames(['test']))
    else:
        test_list = np.load(args.test_list)

    if(args.valid_list is ""):
        val_list = np.array(getListGames(['valid']))
    else:
        val_list = np.load(args.valid_list)

    percent = args.percent
    # random select 100 games from train_list
    train_list_length = len(train_list)
    train_list = train_list[np.random.choice(
        train_list.shape[0], int(train_list_length*percent), replace=False)]
    test_list_length = len(test_list)
    test_list = test_list[np.random.choice(
        test_list.shape[0], int(test_list_length*percent), replace=False)]
    val_list_length = len(val_list)
    val_list = val_list[np.random.choice(
        val_list.shape[0], int(val_list_length*percent), replace=False)]

    # create dataset
    if not args.test_only:
        dataset_Train = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_train,
                                       version=args.version, framerate=args.framerate, window_size=args.window_size, listGames=train_list)
        dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid,
                                       version=args.version, framerate=args.framerate, window_size=args.window_size, listGames=val_list)
        dataset_Test = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_test,
                                        version=args.version, framerate=args.framerate, window_size=args.window_size, listGames=test_list)
                                        
    dataset_Test = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test,
                                         version=args.version, framerate=args.framerate, window_size=args.window_size, listGames=test_list)
    return


if __name__ == '__main__':

    parser = ArgumentParser(description='context aware loss function',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path',   required=False, type=str,
                        default="/path/to/SoccerNet/",     help='Path for SoccerNet')
    parser.add_argument('--features',   required=False, type=str,
                        default="224ppyAudioAnalysis.npy",     help='Video features')
    parser.add_argument('--max_epochs',   required=False, type=int,
                        default=1000,     help='Maximum number of epochs')
    parser.add_argument('--load_weights',   required=False,
                        type=str,   default=None,     help='weights to load')
    parser.add_argument('--model_name',   required=False, type=str,
                        default="NetVLAD++",     help='named of the model to save')
    parser.add_argument('--test_only',   required=False,
                        action='store_true',  help='Perform testing only')

    parser.add_argument('--split_train', nargs='+',
                        default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+',
                        default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+',
                        default=["test", "challenge"], help='list of split for testing')

    parser.add_argument('--version', required=False, type=int,
                        default=2,     help='Version of the dataset')
    parser.add_argument('--feature_dim', required=False, type=int,
                        default=None,     help='Number of input features')
    parser.add_argument('--evaluation_frequency', required=False,
                        type=int,   default=10,     help='Number of chunks per epoch')
    parser.add_argument('--framerate', required=False, type=int,
                        default=2,     help='Framerate of the input features')
    parser.add_argument('--window_size', required=False, type=int,
                        default=15,     help='Size of the chunk (in seconds)')
    parser.add_argument('--pool',       required=False,
                        type=str,   default="NetVLAD++", help='How to pool')
    parser.add_argument('--vocab_size',       required=False, type=int,
                        default=64, help='Size of the vocabulary for NetVLAD')
    parser.add_argument('--NMS_window',       required=False,
                        type=int,   default=30, help='NMS window in second')
    parser.add_argument('--NMS_threshold',       required=False, type=float,
                        default=0.0, help='NMS threshold for positive results')

    parser.add_argument('--batch_size', required=False,
                        type=int,   default=256,     help='Batch size')
    parser.add_argument('--LR',       required=False,
                        type=float,   default=1e-03, help='Learning Rate')
    parser.add_argument('--LRe',       required=False,
                        type=float,   default=1e-06, help='Learning Rate end')
    parser.add_argument('--patience', required=False, type=int,   default=10,
                        help='Patience before reducing LR (ReduceLROnPlateau)')

    parser.add_argument('--GPU',        required=False, type=int,
                        default=-1,     help='ID of the GPU to use')
    parser.add_argument('--max_num_worker',   required=False,
                        type=int,   default=4, help='number of worker to load data')
    parser.add_argument('--seed',   required=False, type=int,
                        default=0, help='seed for reproducibility')

    # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
    parser.add_argument('--loglevel',   required=False,
                        type=str,   default='INFO', help='logging level')
    parser.add_argument('--train_list', required=False,
                        type=str,   default="", help='train list location')
    parser.add_argument('--valid_list', required=False,
                        type=str,   default="", help='valid list location')
    parser.add_argument('--test_list', required=False,
                        type=str,   default="", help='test list location')
    parser.add_argument('--percent', required=False, type=float,
                        default=1.0,     help='percent of data to use')

    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    start = time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
