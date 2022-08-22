import pandas

targets = [
    # 'TestOriginalPooling',
    # # 'TestAvgPooling',
    # # 'TestCBAM'
    # 'TestPLVPooling'
    # 'Test_conv3block3_WithKernel5x5ForTheLast'
    # 'Test_conv3block3_WithKernel3x3ForTheLast'
    # 'StructureTestOnModel_{}',
    # 'TestConvLayers_{}'
    # 'TestFBs_{}'
    # 'TestAvgPooling2'
    # 'TestPLVA2'
    # 'TestMnist'
    'TestCatOrDogv2'
]

# models = ['gdbls_conv2block3', 'gdbls_conv3block2', 'gdbls_conv3block3']
models = ['gdbls_conv3block3_dogcatversion']
# models = [f'gdbls_conv3block{i}' for i in range(1, 5)]
# models = ['resnet_fpn']
for target in targets:
    for model in models:
        for dataset in [
            # 'cifar10', 'cifar100', 'svhn'
            # 'mnist'
            'catordog'
        ]:
            df = pandas.read_csv(f'saves/exp({target})_{model}_on_{dataset}.csv')
            accmean = df.mean(axis=0)
            print(f'{target.format(model)}: gdbls_on_{dataset} mean stat:\n {accmean}')
