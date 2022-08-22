import sys
import logging
import yaml
import requests
import datetime
from ignite_enterance import run as go

# A logger for this file
log = logging.getLogger(__name__)


def gen_tasks(dataset, model, exp_explain):
    """
    Run experiment on certain dataset with DictConfig read from 'configs/[run_dataset].yaml'

    When the program terminates, log files are saved outputs/[day]/[hour-minute-second]
    confusion matrix and checkpoints are saved to logs/[run_dataset]

    options:
        - analyse_time: default False, determines if we should analyse the program progress
        - log_details: default True, determines if we should enable clearml platform & save checkpoints.
        - conclude_train: default Ture, determines analyse confusion matrix and classification report
        - target_model: model name. optional: gdbls, gdbls_conv2block2, gdbls_conv4block4
    """
    func_name = f'task_for_{dataset}_using_{model}'
    log.info(f'Created Task Function {func_name}')

    def run() -> None:
        options = {
            'analyse_time': False,
            'log_details': False,
            'conclude_train': (True if dataset != 'cifar100' else False),
            'target_model': model
        }
        ystr = open(f'configs/{dataset}.yaml', 'r').read()
        cfgdict = yaml.load(ystr, Loader=yaml.FullLoader)
        res = go(cfgdict, options, log)
        json_data = {
            'exp_name': f'exp({exp_explain})_{model}_on_{dataset}',
            'exp_time': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        }
        for key, val in res.items():
            json_data[key] = val
        try:
            r = requests.post("http://127.0.0.1:8003/report", json=json_data)
            print(f'send data:{json_data}')
            print(r.headers)
            print(r.text)
            log.info('PROCESS SUCCESSFULLY TERMINATED.')
        except Exception as e:
            log.error('PROCESS TERMINATED DUE TO BAD CONNECTION. PLZ CHECK IF THE STAT SERVER IS OPEN.')
            print(e)

    globals()[func_name] = run
    run.__name__ = func_name
    return run


if __name__ == '__main__':
    if sys.argv[1] not in ['cifar10', 'cifar100', 'svhn', 'mnist', 'catordog']:
        raise NameError('Not Supported Dataset.')
    if sys.argv[2] not in [
        'gdbls_conv1block3',
        'gdbls_conv2block3',
        'gdbls_conv3block3',
        'gdbls_conv4block3',
        'gdbls_conv5block3',
        'gdbls_conv6block3',

        'gdbls_conv3block1',
        'gdbls_conv3block2',
        'gdbls_conv3block4',

        'gdbls_conv3block3_noEB',
        'resnet_fpn',
        'gdbls_conv3block3_dogcatversion'
    ]:
        raise NameError('Not Provided Model.')
    if sys.argv[3] is None:
        raise AssertionError('Should explain the experiment target')

    gen_tasks(sys.argv[1], sys.argv[2], sys.argv[3])
    globals()[f'task_for_{sys.argv[1]}_using_{sys.argv[2]}']()
