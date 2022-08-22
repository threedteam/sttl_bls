from nni.experiment import Experiment

search_space = {
    # 'batch_size': {'_type': 'randint', '_value': [128, 192]},
    # 'init_lr': {'_type': 'uniform', '_value': [0.0001, 0.001]},
    # 'divns': {'_type': 'randint', '_value': [1, 16]},
    'ppvbias': {'_type': 'uniform', '_value': [-2, 2]},
}

experiment = Experiment('local')
experiment.config.search_space = search_space

experiment.config.trial_command = f'python test_enterance.py cifar10 gdbls nni_tunning'
experiment.config.trial_code_directory = '.'

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {
    'optimize_mode': 'maximize',
    'seed': 42,
}

experiment.config.assessor.name = 'Medianstop'
experiment.config.assessor.class_args = {
    'optimize_mode': 'maximize',
    'start_step': 5
}

experiment.config.max_trial_number = 100
experiment.config.trial_concurrency = 1

experiment.run(8080)

experiment.stop()
