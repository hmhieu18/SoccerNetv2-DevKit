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
        dataset_Train = SoccerNetClips(visual_path=args.SoccerNet_path, audio_path=args.audio_path, visual_features=args.features, audio_feaures=args.audio_feaures, split=args.split_train,
                                       version=args.version, framerate=args.framerate, window_size=args.window_size, listGames=train_list)
        dataset_Valid = SoccerNetClips(visual_path=args.SoccerNet_path, audio_path=args.audio_path, visual_features=args.features, audio_feaures=args.audio_feaures, split=args.split_valid,
                                        version=args.version, framerate=args.framerate, window_size=args.window_size, listGames=val_list)
        # dataset_Valid = SoccerNetClips(path=args.SoccerNet_path, features=args.features, split=args.split_valid,
        #                                       version=args.version, framerate=args.framerate, window_size=args.window_size, listGames=val_list)
    dataset_Test = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test,
                                         version=args.version, framerate=args.framerate, window_size=args.window_size, listGames=test_list)

    if args.visual_feature_dim is None:
        args.visual_feature_dim = dataset_Test[0][1].shape[-1]
        print("visual_feature_dim found:", args.visual_feature_dim)

    if args.audio_feature_dim is None:
        args.audio_feature_dim = dataset_Test[0][1].shape[-1]
        print("visual_feature_dim found:", args.audio_feature_dim)

    # create model
    model = Model(weights=args.load_weights, input_size=args.visual_feature_dim,
                  num_classes=dataset_Test.num_classes, window_size=args.window_size,
                  vocab_size=args.vocab_size,
                  framerate=args.framerate, pool=args.pool).cuda()
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer = [p.numel()
                            for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.max_num_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.max_num_worker, pin_memory=True)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid,
                                                        batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.max_num_worker, pin_memory=True)

    # training parameters
    if not args.test_only:
        criterion = NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR,
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=0, amsgrad=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', verbose=True, patience=args.patience)

        # start training
        trainer(train_loader, val_loader, val_metric_loader,
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency)

    # Free up some RAM memory
    if not args.test_only:
        del dataset_Train, dataset_Valid
        del train_loader, val_loader, val_metric_loader
    else:
        del dataset_Test
    # For the best model only
    checkpoint = torch.load(os.path.join(
        "models", args.model_name, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    # test on multiple splits [test/challenge]
    for split in args.split_test:
        dataset_Test = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test,
                                             version=args.version, framerate=args.framerate, window_size=args.window_size, listGames=test_list)

        test_loader = torch.utils.data.DataLoader(dataset_Test,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=1, pin_memory=True)

        results = testSpotting(test_loader, model=model, model_name=args.model_name,
                               NMS_window=args.NMS_window, NMS_threshold=args.NMS_threshold)
        if results is None:
            continue

        a_mAP = results["a_mAP"]
        a_mAP_per_class = results["a_mAP_per_class"]
        a_mAP_visible = results["a_mAP_visible"]
        a_mAP_per_class_visible = results["a_mAP_per_class_visible"]
        a_mAP_unshown = results["a_mAP_unshown"]
        a_mAP_per_class_unshown = results["a_mAP_per_class_unshown"]

        logging.info("Best Performance at end of training ")
        logging.info("a_mAP visibility all: " + str(a_mAP))
        logging.info("a_mAP visibility all per class: " + str(a_mAP_per_class))
        logging.info("a_mAP visibility visible: " + str(a_mAP_visible))
        logging.info("a_mAP visibility visible per class: " +
                     str(a_mAP_per_class_visible))
        logging.info("a_mAP visibility unshown: " + str(a_mAP_unshown))
        logging.info("a_mAP visibility unshown per class: " +
                     str(a_mAP_per_class_unshown))

    return


if __name__ == '__main__':

    parser = ArgumentParser(description='context aware loss function',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path',   required=False, type=str,
                        default="/path/to/SoccerNet/",     help='Path for SoccerNet')

    parser.add_argument('--audio_path',   required=False, type=str,
                        default="/path/to/SoccerNet/",     help='Path for audio SoccerNet')
    
    parser.add_argument('--features',   required=False, type=str,
                        default="ResNET_TF2.npy",     help='Video features')
    
    parser.add_argument('--audio_features',   required=False, type=str,
                        default="224ppyAudioAnalysis.npy",     help='audio features')
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
    parser.add_argument('--visual_feature_dim', required=False, type=int,
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
