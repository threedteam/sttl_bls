import nni
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from torchvision import transforms, utils
from torch.utils.data import DataLoader

# handel data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit

# ignite tools
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events, Engine
from ignite.metrics import Accuracy, Loss, Metric
from ignite.utils import setup_logger
from ignite.contrib.handlers.clearml_logger import *
from ignite.handlers import EarlyStopping, Checkpoint
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST
from datasets.CATORDOG import DogsVSCatsDataset as CATORDOG
from datasets.ImageNet32.ImageNet32 import ImageNet32

import medmnist
from medmnist import INFO, Evaluator

# analyse tools
from pyinstrument import Profiler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
from thop import profile

# models and data-sets
from model import (
    gdbls_c3b3,
    gdbls_c2b3,
    gdbls_c1b3,
    gdbls_c4b3,
    gdbls_c5b3,
    gdbls_c6b3,
    gdbls_c3b2,
    gdbls_c3b1,
    gdbls_c3b4,
    gdbls_c1b2,
    gdbls_c1b1,
    gdbls_c2b2,
    gdbls_c2b1
)

TissueMnistClass = getattr(medmnist, 'TissueMNIST')
        
class CustomMediClass(TissueMnistClass):
    def __getitem__(self, index):
        d,l = super(CustomMediClass, self).__getitem__(index)
        return d, l.squeeze()

class NewData(Metric):
    def __init__(self, trainer, output_transform=lambda x: x, device="cpu"):
        self.new_data = []
        self.trainer = trainer
        super(NewData, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.new_data = []
        super(NewData, self).reset()

    @reinit__is_reduced
    def update(self, output):
        #return
        y_pred, x = output[2].detach(), output[0].detach()
        softmax = nn.Softmax(dim=1)
        y_pred = softmax(y_pred)
        y_pred = y_pred.cpu().numpy()
        x_test = x.cpu().numpy()
        epoch = self.trainer.state.epoch
        p = 0.9
        for i in range(y_pred.shape[0]):
            p_max = y_pred[i].max()
            label = np.argmax(y_pred[i])
            if p_max >= p:
                data = [x_test[i], label]
                self.new_data.append(data)

    @sync_all_reduce("new_data")
    def compute(self):
        data = np.asarray(self.new_data)
        return data

    # def iteration_completed(self, engine: Engine) -> None:
    #     output = self._output_transform(engine.state.output)
    #     if isinstance(output, Mapping):
    #         if self.required_output_keys is None:
    #             raise TypeError(
    #                 f"Transformed engine output for {self.__class__.__name__} metric should be a tuple/list, "
    #                 f"but given {type(output)}"
    #             )
    #         if not all([k in output for k in self.required_output_keys]):
    #             raise ValueError(
    #                 "When transformed engine's output is a mapping, "
    #                 f"it should contain {self.required_output_keys} keys, but given {list(output.keys())}"
    #             )
    #         output = tuple(output[k] for k in self.required_output_keys)
    #     self.update(output, engine.state.epoch)
    #     engine.t

class CustomSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.new_data = []
        self.new_data_len = 0

    def __getitem__(self, idx):
        if idx >= len(self.indices):
            return torch.from_numpy(self.new_data[idx - len(self.indices)][0]), self.new_data[idx - len(self.indices)][1]
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices) + self.new_data_len

    def add(self, new_data):
        self.new_data_len = new_data.shape[0]
        self.new_data = np.asarray(new_data)
        for i in range(self.new_data_len):
            self.new_data[i][0] = np.float32(self.new_data[i][0])

def get_data(config, logger):
    dataset_name = config['dataset_name']

    if dataset_name in ['MNIST', 'TISSUEMNIST']:
        mean = tuple(config['mean'])
        std = tuple(config['std'])
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset_name == 'CATORDOG':
        IMAGE_SIZE = 64
        # 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式
        transform_train = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),  # 将图像按比例缩放至合适尺寸
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 从图像中心裁剪合适大小的图像
            transforms.ToTensor(),  # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]，同时将H×W×C的数据转置成C×H×W，这一点很关键
        ])
        transform_test = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),  # 将图像按比例缩放至合适尺寸
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 从图像中心裁剪合适大小的图像
            transforms.ToTensor(),  # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]，同时将H×W×C的数据转置成C×H×W，这一点很关键
        ])
    elif dataset_name == 'IMAGENET32':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(p=0.5)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        mean = tuple(config['mean'])
        std = tuple(config['std'])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(p=0.5)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST']:
        trainset = eval(dataset_name)(root='datasets/' + dataset_name, train=True, download=True,
                                      transform=transform_train)
        validset = eval(dataset_name)(root='datasets/' + dataset_name, train=True, download=True,
                                      transform=transform_test)
        testset = eval(dataset_name)(root='datasets/' + dataset_name, train=False, download=True,
                                     transform=transform_test)

    elif dataset_name == 'CATORDOG':
        trainset = CATORDOG(root='datasets/CATORDOG', mode='train',
                            transform=transform_train)
        validset = CATORDOG(root='datasets/CATORDOG', mode='train',
                            transform=transform_test)
        testset = CATORDOG(root='datasets/CATORDOG', mode='test',
                           transform=transform_test)
    elif dataset_name == 'SVHN':
        trainset = SVHN(root='datasets/' + dataset_name, download=True, split='train',
                        transform=transform_train)
        validset = SVHN(root='datasets/' + dataset_name, download=True, split='train',
                        transform=transform_test)
        testset = SVHN(root='datasets/' + dataset_name, download=True, split='test',
                       transform=transform_test)
    elif dataset_name == 'IMAGENET32':
        train_data_folder = 'datasets/ImageNet32/Imagenet32_train'
        trainset = validset = ImageNet32(train_data_folder, transform_train, batchs=10)
        test_data_folder = 'datasets/ImageNet32/Imagenet32_val'
        testset = ImageNet32(test_data_folder, transform_test, batchs=0)
    elif dataset_name == 'TISSUEMNIST':
        trainset = CustomMediClass(split='train', transform=transform_train, download=True)
        validset = CustomMediClass(split='val', transform=transform_train, download=True)
        testset = CustomMediClass(split='test', transform=transform_test, download=True)

    if dataset_name == 'CIFAR10':
        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == 'CIFAR100':
        label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                       'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                       'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
                       'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
                       'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                       'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
                       'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                       'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                       'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                       'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
                       'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
                       'willow_tree', 'wolf', 'woman', 'worm']
    elif dataset_name in ['MNIST', 'SVHN']:
        label_names = [str(i) for i in range(10)]
    elif dataset_name == 'CATORDOG':
        label_names = ['cat', 'dog']
    elif dataset_name == 'IMAGENET32':
        label_names = []
        with open('datasets/ImageNet32/label_names.txt', 'r') as f:
            for line in f.readlines():
                label_names.append(line.strip())
    elif dataset_name == 'TISSUEMNIST':
        label_names = ['Collecting Duct, Connecting Tubule', 'Distal Convoluted Tubule', 'Glomerular endothelial cells', 'Interstitial endothelial cells', 'Leukocytes', 'Podocytes', 'Proximal Tubule Segments', 'Thick Ascending Limb']

    
    if dataset_name == 'IMAGENET32':
        # Select 20 samples from each of the 1000 classes for validation
        val_size = 20
        val_indices = []
        for i in range(1000):
            val_indices.extend(trainset.get_class_indices(i)[:val_size])
        train_indices = list(set(range(len(trainset))) - set(val_indices))
        
        # Define data samplers for train and validation sets
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # Define dataloaders
        trainloader = DataLoader(trainset, batch_size=config['cfg']["batch_size"], sampler=train_sampler, num_workers=4, drop_last=True, pin_memory=True)
        validloader = DataLoader(trainset, batch_size=config['cfg']["batch_size"], sampler=val_sampler, num_workers=4, drop_last=True, pin_memory=True)
        testloader = DataLoader(testset, batch_size=config['cfg']["batch_size"], num_workers=4, drop_last=True, pin_memory=True)
    
    elif dataset_name == 'TISSUEMNIST':
        trainloader = DataLoader(dataset=trainset, batch_size=config['cfg']["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
        validloader = DataLoader(dataset=validset, batch_size=config['cfg']["batch_size"], shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
        testloader = DataLoader(dataset=testset, batch_size=config['cfg']["batch_size"], shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
        
        
    elif config['cfg']['test_size'] != 0:
        labels = [trainset[i][1] for i in range(len(trainset))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=config['cfg']['test_size'])
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]

        trainset = CustomSubset(trainset, train_indices)
        validset = torch.utils.data.Subset(validset, valid_indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['cfg']["batch_size"],
                                                  shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        validloader = torch.utils.data.DataLoader(validset, batch_size=config['cfg']["batch_size"],
                                                  shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['cfg']["batch_size"],
                                                 shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['cfg']["batch_size"],
                                                  shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        validloader = testloader = torch.utils.data.DataLoader(testset, batch_size=config['cfg']["batch_size"],
                                                               shuffle=True, drop_last=True, num_workers=4,
                                                               pin_memory=True)
    for X, y in testloader:
        logger.info(f"load data complete:")
        logger.info(f"data area: [{X.min()},{X.max()}]")
        logger.info(f"Shape of X [N, C, H, W]: {X.shape}")
        logger.info(f"Shape of y [N, label]: {y.shape} {y.dtype}")
        break

    return trainloader, validloader, testloader, label_names, trainset, testset

def log_FLOPs_Params(model, example_input):
    flops, params = profile(
        model, inputs=(example_input,)
    )  # 计算模型的FLOPs和参数数量
    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")  # 打印FLOPs，单位为GFLOPs
    print(f"Params: {params / 1e6:.3f} M")  # 打印参数数量，单位为百万(M)
    return flops / 1e9, params / 1e6

def score_function(engine):
    val_loss = engine.state.metrics['loss']
    # val_acc = engine.state.metrics['accuracy']
    return -val_loss

model_stat = dict(
    FLOPS=None,
    param=None
)

def run(config, options, logger):
    timeseries = []
    test_timeseries = []
    acc, loss = [], []
    train_losses = []
    val_losses = []
    final_test_time = []

    torch.manual_seed(42)

    params = config['cfg']
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")
    torch.backends.cudnn.benchmark = True

    # we should load data first exactly.
    assert get_data is not None
    train_loader, val_loader, test_dataloader, label_names, trainset, testset = get_data(config, logger)
    if options['target_model'] == 'mobileformer':
        model = eval(options['target_model']).GDBLS()
    else:
        model = eval(options['target_model']).GDBLS(
            num_classes=config['num_classes'],
            input_shape=config['input_shape'],
            overall_dropout=params["overall_dropout"],
            filters=params["filters"],
            divns=[params["divns"], params["divns"], params["divns"], params["divns"]],
            dropout_rate=params["dropout_rate"],
            batchsize=params["batch_size"],
        )
    ex_array = torch.zeros(
        (params["batch_size"], *config['input_shape']),
        dtype=torch.float32,
    )
    print(ex_array.shape)
    flops, param = log_FLOPs_Params(model, ex_array)
    model_stat['FLOPS']=flops
    model_stat['param']=param
    print(f'About the current Model: {model_stat}')
    
    model.cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['cfg']['init_lr'], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    lrpatience = 3
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lrpatience, verbose=True, factor=0.2)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    trainer.logger = setup_logger("trainer")

    metrics = {"accuracy": Accuracy(), "loss": Loss(loss_fn)}

    # for training eval
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger("Train Evaluator")

    # for validation eval
    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator.logger = setup_logger("Val Evaluator")

    metrics = {"accuracy": Accuracy(output_transform=lambda x: (x[2], x[1])),
               "loss": Loss(loss_fn, output_transform=lambda x: (x[2], x[1])), "new_data": NewData(trainer)}

    test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device,
                                                       output_transform=lambda x, y, y_pred: (x, y, y_pred))
    test_evaluator.logger = setup_logger("Test Evaluator")


    early_stop_patience = 10
    early_stop_handler = EarlyStopping(patience=early_stop_patience, score_function=score_function, trainer=trainer, min_delta=0.005)
    validation_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)  # set early stopping handler

    clearml_logger = None
    if options['log_details']:
        # To utilize other loggers we need to change the object here
        clearml_logger = ClearMLLogger(project_name="examples", task_name="ignite")

        # Attach the logger to the trainer to log training loss
        clearml_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED(every=100),
            tag="training",
            output_transform=lambda loss: {"batchloss": loss},
        )
        # Attach the logger to log loss and accuracy for both training and validation
        for tag, evaluator in [("training metrics", train_evaluator), ("validation metrics", validation_evaluator)]:
            clearml_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )
        # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate
        clearml_logger.attach_opt_params_handler(
            trainer, event_name=Events.EPOCH_COMPLETED(every=1), optimizer=optimizer
        )

        # # Attach the logger to the trainer to log model's weights as a scalar
        # clearml_logger.attach(trainer, log_handler=WeightsScalarHandler(model),
        #                       event_name=Events.EPOCH_COMPLETED(every=1))
        # # Attach the logger to the trainer to log model's gradients as a histogram
        # clearml_logger.attach(trainer, log_handler=GradsScalarHandler(model),
        #                       event_name=Events.EPOCH_COMPLETED(every=1))

        # save the best checkpoint
        handler = Checkpoint(
            {"model": model},
            ClearMLSaver(dirname=config['log_pth'] + '/saves', require_empty=False),
            n_saved=1,
            score_function=lambda e: e.state.metrics["accuracy"],
            score_name="val_acc",
            filename_prefix="best",
            global_step_transform=global_step_from_engine(trainer),
        )
        validation_evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)  # set Checkpoint handler

    @trainer.on(Events.EPOCH_STARTED)
    def print_lr():
        lr = optimizer.param_groups[0]["lr"]
        trainer.logger.info(f"Epoch[{trainer.state.epoch}]: Current learning rate is {lr}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        train_evaluator.run(train_loader)
        logger.info(f'Train evaluate result: {train_evaluator.state.metrics}')
        train_metrics = train_evaluator.state.metrics
        train_loss = train_metrics["loss"]
        train_losses.append(train_loss)

        # if not using STTL, use these
        # validation_evaluator.run(val_loader)
        # logger.info(f'Validation evaluate result: {validation_evaluator.state.metrics["accuracy"]}')

        # test_timeseries.append(validation_evaluator.state.times[validation_evaluator.last_event_name.name])

        # lr_scheduler.step(validation_evaluator.state.metrics["loss"])

        # nni.report_intermediate_result(validation_evaluator.state.metrics["accuracy"])

        # torch.cuda.empty_cache()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_time(engine):
        logger.info(f"{trainer.last_event_name.name} took {trainer.state.times[trainer.last_event_name.name]} seconds")
        timeseries.append(trainer.state.times[trainer.last_event_name.name])

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_dataset(engine):
    
        validation_evaluator.run(val_loader)
        logger.info(f'Validation evaluate result: {validation_evaluator.state.metrics}')
        validation_loss = (-1) * validation_evaluator.state.metrics["loss"]
    
        test_timeseries.append(validation_evaluator.state.times[validation_evaluator.last_event_name.name])
    
        lr_scheduler.step(validation_evaluator.state.metrics["loss"])
    
        nni.report_intermediate_result(validation_evaluator.state.metrics["accuracy"])
    
        val_metrics = validation_evaluator.state.metrics
        val_loss = val_metrics["loss"]
        val_losses.append(val_loss)
    
        torch.cuda.empty_cache()
    
        # run test dataset
        test_evaluator.run(test_dataloader)
        metrics = test_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        new_data = metrics["new_data"]
        logger.info(
            f"Test Results - Avg accuracy: {avg_accuracy:.4f} Avg loss: {avg_loss:.4f}"
        )

        # STTL
        logger.info(
            f"New Data Shape: {new_data.shape} "
        )
        temp_trainset = copy.deepcopy(trainset)
        temp_trainset.add(new_data)
        trainloader = torch.utils.data.DataLoader(temp_trainset, batch_size=config['cfg']["batch_size"],
                                                  shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        trainer.set_data(trainloader)


    @trainer.on(Events.COMPLETED)
    def do_test(engine):
        if options['log_details']:
            clearml_logger.close()

        # run test dataset
        validation_evaluator.run(test_dataloader)
        metrics = validation_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        logger.info(
            f"Done, Test Results - Avg accuracy: {avg_accuracy:.4f} Avg loss: {avg_loss:.4f}"
        )
        final_test_time.append(validation_evaluator.state.times[
                                   validation_evaluator.last_event_name.name
                               ])
        nni.report_final_result(avg_accuracy)

        # if options['conclude_train']:
        #     y_pred = validation_evaluator.state.output[0].cpu().numpy()
        #     y_pred = np.argmax(y_pred, axis=1)
        #     y_true = validation_evaluator.state.output[1].cpu().numpy()
        #
        #     # draw confusion matrix
        #     cf_matrix = confusion_matrix(y_pred, y_true, normalize='true')
        #     df_cm = pd.DataFrame(cf_matrix, index=label_names, columns=label_names)
        #     plt.figure(figsize=(20, 20))
        #     sn.heatmap(df_cm, annot=True)
        #     plt.savefig(config['log_pth'] + '/confusion_matrix.png')
        #
        #     # print classification report
        #     logger.info(f"Classification report:{classification_report(y_true, y_pred, target_names=label_names)}")
        acc.append(f'{avg_accuracy:.4f}')
        loss.append(f'{avg_loss:.4f}')
        torch.cuda.empty_cache()

        if options['log_details']:
            clearml_logger.close()

    if options['analyse_time']:  # use pyinsnstrument profiler to analyse performance in time
        profiler = Profiler()
        profiler.start()
        trainer.run(train_loader, max_epochs=config['cfg']["epochs"])
        profiler.stop()
        logger.info(profiler.output_text(unicode=True, color=True))
    else:
        trainer.run(train_loader, max_epochs=config['cfg']["epochs"])

    logger.info(f'Training time cost: {sum(timeseries):.3f}')
    logger.info(f'Average Training time cost Per Epoch: {sum(timeseries) / len(timeseries):.3f}')
    logger.info(f'Average Validation time cost Per Epoch: {sum(test_timeseries) / len(test_timeseries):.3f}')

    # plot loss
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.plot(np.arange(len(train_losses))+1, train_losses, label="train")
    plt.plot(np.arange(len(val_losses))+1, val_losses, label="validation")

    plt.ylim(0.5, 4.2)
    plt.legend()  # 显示图例
    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.show()
    imgName = config['dataset_name'] + '.png'
    plt.savefig(imgName, dpi=1000)

    tlName = config['dataset_name'] + '_trainloss.csv'
    vlName = config['dataset_name'] + '_valloss.csv'
    np.savetxt(tlName, train_losses, delimiter=',')
    np.savetxt(vlName, val_losses, delimiter=',')

    return {
        'train_time': float(f'{sum(timeseries):.3f}'),
        'train_time_epoch': float(f'{sum(timeseries) / len(timeseries):.3f}'),
        'val_time_epoch': float(f'{sum(test_timeseries) / len(test_timeseries):.3f}'),
        'last_test_time_consumption': float(final_test_time[0]),
        'acc': float(acc[0]),
        'loss': float(loss[0]),
        **model_stat
    }
