"""eval resnet."""
import os
import argparse
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.CrossEntropySmooth import CrossEntropySmooth

'''
5个参数：
--net 网络结构 默认 resnet50
--dataset 数据集 默认 rp2k
--checkpoint_path 
--dataset_path 测试集路径
--device_target 设备 'Ascend'/'CPU'/'GPU'
'''
parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--net', type=str, default='resnet50', help='Resnet Model, either resnet50 or resnet101')
parser.add_argument('--dataset', type=str, default='rp2k', help='Dataset, either cifar10 or imagenet2012 or rp2k')

parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
args_opt = parser.parse_args()

set_seed(1)


if args_opt.net == "resnet50":
    from src.resnet import resnet50 as resnet
    from src.config import config5 as config
    from src.dataset import create_dataset5 as create_dataset

'''
if args_opt.net == "resnet50":
    from src.resnet import resnet50 as resnet
    if args_opt.dataset == "cifar10":
        from src.config import config1 as config
        from src.dataset import create_dataset1 as create_dataset
    else:
        from src.config import config2 as config
        from src.dataset import create_dataset2 as create_dataset
elif args_opt.net == "resnet101":
    from src.resnet import resnet101 as resnet
    from src.config import config3 as config
    from src.dataset import create_dataset3 as create_dataset
else:
    from src.resnet import se_resnet50 as resnet
    from src.config import config4 as config
    from src.dataset import create_dataset4 as create_dataset
'''

if __name__ == '__main__':
    target = args_opt.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size,
                             target=target)
    step_size = dataset.get_dataset_size()
    print('output_shapes: %s', dataset.output_shapes())

    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    if args_opt.dataset == "imagenet2012" or args_opt.dataset == "rp2k":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction='mean',
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
