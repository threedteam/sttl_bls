import os

if __name__ == '__main__':
    datasets = [
        # 'cifar10',
        # 'cifar100',
        # 'svhn'
        # 'mnist',
        'catordog'
    ]
    models = [
        # f'gdbls_conv3block{i}' for i in range(1, 5)
        'gdbls_conv3block3_dogcatversion'
        # 'resnet_fpn'
    ]
    print(f'models to test: {models}')
    # target = 'TestOriginalPooling'
    # target = 'TestAvgPooling'
    # target = 'TestCBAM'
    # target = 'TestPLVPooling'
    # target = 'Test_conv3block3_WithKernel5x5ForTheLast'
    # target = 'Test_conv3block3_WithKernel3x3ForTheLast'
    # target = 'StructureTestOnModel_{}'
    # target = 'WithoutDropout_{}'
    # target = 'TestConvLayers_{}'
    # target = 'TestFBs_{}'
    # target = 'TestKaiMingUniformInit_{}'
    # target = 'TestDifferentInitSeed3407'
    # target = 'resnet_fpnTest_earlystoppatience40_lrpatience20_drop03'
    # target = 'TestAvgPooling2'
    # target = 'TestPLVA2'
    # target = 'TestMnist'
    target = 'TestCatOrDogv2'
    
    repeat = 5
    for i in range(repeat):
        for dataset in datasets:
            for model in models:
                os.system(f'python test_enterance.py {dataset} {model} {target.format(model)}')
