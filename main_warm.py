import torch
from Node import Node, Global_Node
from Args import args_parser
from Data import Data
from utils import LR_scheduler, Recorder, exp_details, Summary, dimension_reduction
from Trainer import Trainer, train_classifier, train_fc, train_ssl
from log import logger_config, set_random_seed
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter


# init args
args = args_parser()

comments = f"{args.dataset}-r{args.R}-lr{args.lr}-le{args.E}-bs{args.batch_size}-alpha{args.alpha}-beta{args.beta}-it{args.iteration}-{args.algorithm}"
print(comments)
result_name = str(datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-", "_")+'_'+comments

curr_dir = './runs/' + result_name
if not os.path.exists(curr_dir):
    os.makedirs(curr_dir)

summary_writer = SummaryWriter(log_dir=curr_dir, comment=comments)



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
            Train(Node_List[k],args,logger,rounds,summary_writer, epoch)

        if args.algorithm == 'fed_adv':
            train_ssl(Node_List[k], args, logger, rounds, summary_writer)
            # train_classifier(Node_List[k], args, logger, rounds, summary_writer)
            # recorder.validate(Node_List[k], summary_writer)
            recorder.test_on_target(Node_List[k], summary_writer, rounds)
            if rounds == args.R-1:
                dimension_reduction(Node_List[k], Data, rounds)
        elif args.algorithm == 'fed_mutual':
            recorder.printer(Node_List[k])
            Global_node.fork(Node_List[k])
            recorder.printer(Global_node)
            recorder.validate(Global_node, summary_writer)
        elif args.algorithm == 'fed_avg':
            recorder.validate(Node_List[k], summary_writer)
            recorder.test_on_target(Node_List[k], summary_writer, rounds)
    
    if args.algorithm == 'fed_adv':
        acc_list = []
        # for node in Node_List:
        #     acc_list.append(recorder.target_acc[str(node.num)][-1])
        Global_node.merge_weights_gen(Node_List, acc_list)
        for n_ in range(len(Node_List)):
            Node_List[n_].local_fork_gen(Global_node)
            train_fc(Node_List[n_], args, logger, rounds, summary_writer)

        proto = Global_node.aggregate(Node_List)
        Global_node.merge_weights_ssl(Node_List, acc_list)
        # TODO 在服务器上利用target数据进行simclr对比学习
        # Global_node.train(rounds, logger, summary_writer)
        
        recorder.server_test_on_target(Global_node, summary_writer, rounds)
        for k_ in range(len(Node_List)):
            Node_List[k_].fork_proto(proto)
            Node_List[k_].local_fork_ssl(Global_node)
            recorder.validate(Node_List[k_], summary_writer)
            
        logger.info(f"iter: {args.iteration}, epoch: {rounds}")

    elif args.algorithm != 'fed_adv':
        if args.algorithm == 'fed_avg':
            Global_node.merge(Node_List)
        elif args.algorithm == 'fed_mutual':
            logger.info("iteration:{},epoch:{},accurancy:{},loss:{}".format(args.iteration, rounds, recorder.log(Global_node)[0], recorder.log(Global_node)[1]))
    # TODO 为fedavg写一个server test on target的函数
    if args.algorithm == 'fed_adv' and rounds == args.R-1:
        dimension_reduction(Global_node, Data, rounds)
recorder.finish()
Summary(args, logger)
logger.info(result_name)