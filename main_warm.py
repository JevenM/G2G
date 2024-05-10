import torch
from Node import Node, Global_Node
from Args import args_parser
from Data import Data
from utils import LR_scheduler, Recorder, exp_details, Summary, dimension_reduction
from Trainer import Trainer
from log import logger_config, set_random_seed
from datetime import datetime
import os

# init args
args = args_parser()

comments = f"{args.dataset}-r{args.R}-lr{args.lr}-le{args.E}-bs{args.batch_size}-alpha{args.alpha}-beta{args.beta}-it{args.iteration}-{args.algorithm}"
print(comments)
result_name = str(datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-", "_")+'_'+comments

# curr_working_dir = os.getcwd()
save_dir = os.path.join('./logger/', result_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

args.save_path = save_dir

images_path = os.path.join(save_dir, 'gen_images')
log_name = os.path.join(save_dir, 'train.log')
if not os.path.exists(images_path):
    os.makedirs(images_path)

save_model_path = os.path.join(save_dir, 'save/model')
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
save_record_path = os.path.join(save_dir, 'save/record')
if not os.path.exists(save_record_path):
    os.makedirs(save_record_path)



date_t = str(datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-", "_")
logger = logger_config(log_path=log_name, logging_name=args.algorithm)
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
logger.info(f'Running on {args.device}')
logger.info(result_name)
exp_details(args, logger)

Data = Data(args, logger)


# init nodes
Global_node = Global_Node(Data.target_loader, args)
Node_List = [Node(k, Data.train_loader[k], Data.test_loader[k], args, Data.target_loader) for k in range(args.node_num)]
# Catfish(Node_List, args)
logger.info(f"Node_list.size = {len(Node_List)}")

recorder = Recorder(args,logger)
Summary(args, logger)
# start
Train = Trainer(args)
for rounds in range(args.R):
    logger.info('===============The {:d}-th round==============='.format(rounds + 1))
    # if args.lr_scheduler == True:
    #     LR_scheduler(rounds, Node_List, args, logger=logger)
    for k in range(len(Node_List)):
        if args.algorithm != 'fed_adv': 
            Node_List[k].fork(Global_node)
        for epoch in range(args.E):
            Train(Node_List[k],args,logger,rounds)
        if args.algorithm == 'fed_adv':
            recorder.validate(Node_List[k])
            recorder.test_on_target(Node_List[k])
        else:
            recorder.printer(Node_List[k])
            Global_node.fork(Node_List[k])
            recorder.printer(Global_node)
            recorder.validate(Global_node)
        
        if args.algorithm == 'fed_adv' and rounds == args.R-1:
            dimension_reduction(Node_List[k], Data, rounds)
    if args.algorithm == 'fed_adv':
        proto = Global_node.aggregate(Node_List)
        for k_ in range(len(Node_List)):
            Node_List[k_].fork_proto(proto)
        logger.info(f"iter: {args.iteration}, epoch: {rounds}")
    else:
        logger.info("iteration:{},epoch:{},accurancy:{},loss:{}".format(args.iteration, rounds, recorder.log(Global_node)[0], recorder.log(Global_node)[1]))
recorder.finish()
Summary(args, logger)
logger.info(result_name)