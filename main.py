import os
import torch, gc

if __name__ == '__main__':
    datasets = [
        'cifar100',
        'cifar10',
        'svhn',
        'mnist',
        'catordog',
        'ImageNet32',
        'TissueMnist'
    ]
    models = [
        'gdbls_c3b3',
        'gdbls_c2b3',
        'gdbls_c1b3',
        'gdbls_c4b3',
        'gdbls_c5b3',
        'gdbls_c6b3',
        'gdbls_c3b2',
        'gdbls_c3b1',
        'gdbls_c3b4',
        'gdbls_c1b2',
        'gdbls_c1b1',
        'gdbls_c2b2',
        'gdbls_c2b1'
        
    ]
    print(f'models to test: {models}')
    # target = 'Testc3b3_former'
    # target = 'Testc2b3_former'
    # target = 'Testc1b3_former'
    # target = 'Testc4b3_former'
    # target = 'Testc5b3_former'
    # target = 'Testc6b3_former'
    # target = 'Testc3b2_former'
    # target = 'Testc3b1_former'
    # target = 'Testc3b4_former'
    # target = 'Testc1b1_on_huge_datasets'
    # target = 'Testc2b1_on_huge_datasets'
    # target = 'Testc1b2_on_huge_datasets'
    # target = 'Testc5b3_on_huge_datasets'
    # target = 'Testc3b2_on_huge_datasets'
    # target = 'TestOtherModels_on_huge_datasets'

    for dataset in datasets:
        for model in models:
            for i in range(1):
                os.system(f'python test_enterance.py {dataset} {model} {target.format(model)}')
