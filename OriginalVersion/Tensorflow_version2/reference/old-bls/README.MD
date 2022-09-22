# Grand descent board learning system implementation

## FIRSTLY: how to use:

- the project is built on CUDA11.4 and you should keep the latest versions of tf and keras.

it's easy to launch a process of learning:
```shell script
python run.py [dataset name]
```
dataset name can be chose from cifar10, cifar100, svhn in the current repo, or you can custom a new config file by doing as follows:
1. create a model file named with the target dataset name in the ./model folder using the template of cifar10.py
2. create config file in the form of cifar10 in ./configs file named with the target dataset.
    ATTENTION: you must change the experiment file in the config file otherwise the record file will be overwritten.
    
3. run it in this way:
    ```shell script
    python run.py cifar10
    ```

## SECONDLY: introduction to gdbls
GDBLS is a implementation of bls in the from of cnn. The original bls is limited by the poor efficiency of feature extracting of human designed extractor, so we 
proposed another way of implementing bls by taking the advantage of cnn to extract the target feature more effectively.

The simple introduction to this idea can be described as follows:

- cnn based feature extractor
- board system implemented by deepwise way
- flexible model hierarchy using feature blocks and changeable final discriminator.
- using adamGard to train the whole network.

## THIRDLY: tools provided for developing the model
We provided some useful tools to facilitate further development of GDBLS.

- grid search: use optimizer.py as follows:
    ```shell script
    python optimizer.py cifar10
    ```
  exactly, you should firstly choose series of attributes in the template function gen_grid() as follows:
    ```python
    def gen_grid():
        # 10 times
        lr_decay_steps = [i for i in range(100000, 200000, 10000)]
        # 4 times
        filters = [[i, i, i] for i in range(128, 192, 16)]
    
        opts = []
        for i in lr_decay_steps:
            for j in filters:
                opts.append(dict(lr_decay_steps=i, filters=j))
        return opts
    ```
  We selected learning rate decay step and number of output channels as target of grid searching, 
  and the search process will begin right after you entered the optimize command.
  
  ATTENTION: this tool really needs much RAM so please ensure that there is only one model running on your machine.
  
- learning process visualization (this is tensorboard tool): 
    ```shell script
    tensorboard --logdir=./tensorboards
    ```

## LASTLY: example training outcome
```text
2022-04-04 21:30:07.591877: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 10283 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 32, 32, 3)]  0           []                               
                                                                                                  
 conv2d (Conv2D)                (None, 32, 32, 86)   2408        ['input_1[0][0]']                
                                                                                                  
 batch_normalization (BatchNorm  (None, 32, 32, 86)  344         ['conv2d[0][0]']                 
 alization)                                                                                       
                                                                                                  
 dropout (Dropout)              (None, 32, 32, 86)   0           ['batch_normalization[0][0]']    
                                                                                                  
 conv2d_1 (Conv2D)              (None, 32, 32, 86)   66650       ['dropout[0][0]']                
                                                                                                  
 batch_normalization_1 (BatchNo  (None, 32, 32, 86)  344         ['conv2d_1[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 dropout_1 (Dropout)            (None, 32, 32, 86)   0           ['batch_normalization_1[0][0]']  
                                                                                                  
 conv2d_2 (Conv2D)              (None, 32, 32, 172)  133300      ['dropout_1[0][0]']              
                                                                                                  
 batch_normalization_2 (BatchNo  (None, 32, 32, 172)  688        ['conv2d_2[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 tf.math.greater (TFOpLambda)   (None, 32, 32, 172)  0           ['batch_normalization_2[0][0]']  
                                                                                                  
 tf.cast (TFOpLambda)           (None, 32, 32, 172)  0           ['tf.math.greater[0][0]']        
                                                                                                  
 tf.math.reduce_mean (TFOpLambd  (None, 172)         0           ['tf.cast[0][0]']                
 a)                                                                                               
                                                                                                  
 dense (Dense)                  (None, 172)          29756       ['tf.math.reduce_mean[0][0]']    
                                                                                                  
 dense_1 (Dense)                (None, 172)          29756       ['dense[0][0]']                  
                                                                                                  
 reshape (Reshape)              (None, 1, 1, 172)    0           ['dense_1[0][0]']                
                                                                                                  
 multiply (Multiply)            (None, 32, 32, 172)  0           ['batch_normalization_2[0][0]',  
                                                                  'reshape[0][0]']                
                                                                                                  
 max_pooling2d (MaxPooling2D)   (None, 16, 16, 172)  0           ['multiply[0][0]']               
                                                                                                  
 dropout_2 (Dropout)            (None, 16, 16, 172)  0           ['max_pooling2d[0][0]']          
                                                                                                  
 conv2d_3 (Conv2D)              (None, 16, 16, 86)   133214      ['dropout_2[0][0]']              
                                                                                                  
 batch_normalization_3 (BatchNo  (None, 16, 16, 86)  344         ['conv2d_3[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 dropout_3 (Dropout)            (None, 16, 16, 86)   0           ['batch_normalization_3[0][0]']  
                                                                                                  
 conv2d_4 (Conv2D)              (None, 16, 16, 86)   66650       ['dropout_3[0][0]']              
                                                                                                  
 batch_normalization_4 (BatchNo  (None, 16, 16, 86)  344         ['conv2d_4[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 dropout_4 (Dropout)            (None, 16, 16, 86)   0           ['batch_normalization_4[0][0]']  
                                                                                                  
 conv2d_5 (Conv2D)              (None, 16, 16, 172)  133300      ['dropout_4[0][0]']              
                                                                                                  
 batch_normalization_5 (BatchNo  (None, 16, 16, 172)  688        ['conv2d_5[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 tf.math.greater_1 (TFOpLambda)  (None, 16, 16, 172)  0          ['batch_normalization_5[0][0]']  
                                                                                                  
 tf.cast_1 (TFOpLambda)         (None, 16, 16, 172)  0           ['tf.math.greater_1[0][0]']      
                                                                                                  
 tf.math.reduce_mean_1 (TFOpLam  (None, 172)         0           ['tf.cast_1[0][0]']              
 bda)                                                                                             
                                                                                                  
 dense_2 (Dense)                (None, 86)           14878       ['tf.math.reduce_mean_1[0][0]']  
                                                                                                  
 dense_3 (Dense)                (None, 172)          14964       ['dense_2[0][0]']                
                                                                                                  
 reshape_1 (Reshape)            (None, 1, 1, 172)    0           ['dense_3[0][0]']                
                                                                                                  
 multiply_1 (Multiply)          (None, 16, 16, 172)  0           ['batch_normalization_5[0][0]',  
                                                                  'reshape_1[0][0]']              
                                                                                                  
 max_pooling2d_1 (MaxPooling2D)  (None, 8, 8, 172)   0           ['multiply_1[0][0]']             
                                                                                                  
 dropout_5 (Dropout)            (None, 8, 8, 172)    0           ['max_pooling2d_1[0][0]']        
                                                                                                  
 conv2d_6 (Conv2D)              (None, 8, 8, 86)     133214      ['dropout_5[0][0]']              
                                                                                                  
 batch_normalization_6 (BatchNo  (None, 8, 8, 86)    344         ['conv2d_6[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 dropout_6 (Dropout)            (None, 8, 8, 86)     0           ['batch_normalization_6[0][0]']  
                                                                                                  
 conv2d_7 (Conv2D)              (None, 8, 8, 86)     66650       ['dropout_6[0][0]']              
                                                                                                  
 batch_normalization_7 (BatchNo  (None, 8, 8, 86)    344         ['conv2d_7[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 dropout_7 (Dropout)            (None, 8, 8, 86)     0           ['batch_normalization_7[0][0]']  
                                                                                                  
 conv2d_8 (Conv2D)              (None, 8, 8, 172)    369972      ['dropout_7[0][0]']              
                                                                                                  
 batch_normalization_8 (BatchNo  (None, 8, 8, 172)   688         ['conv2d_8[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 tf.math.greater_2 (TFOpLambda)  (None, 8, 8, 172)   0           ['batch_normalization_8[0][0]']  
                                                                                                  
 tf.cast_2 (TFOpLambda)         (None, 8, 8, 172)    0           ['tf.math.greater_2[0][0]']      
                                                                                                  
 tf.math.reduce_mean_2 (TFOpLam  (None, 172)         0           ['tf.cast_2[0][0]']              
 bda)                                                                                             
                                                                                                  
 dense_4 (Dense)                (None, 43)           7439        ['tf.math.reduce_mean_2[0][0]']  
                                                                                                  
 dense_5 (Dense)                (None, 172)          7568        ['dense_4[0][0]']                
                                                                                                  
 reshape_2 (Reshape)            (None, 1, 1, 172)    0           ['dense_5[0][0]']                
                                                                                                  
 multiply_2 (Multiply)          (None, 8, 8, 172)    0           ['batch_normalization_8[0][0]',  
                                                                  'reshape_2[0][0]']              
                                                                                                  
 max_pooling2d_2 (MaxPooling2D)  (None, 4, 4, 172)   0           ['multiply_2[0][0]']             
                                                                                                  
 dropout_8 (Dropout)            (None, 4, 4, 172)    0           ['max_pooling2d_2[0][0]']        
                                                                                                  
 flatten (Flatten)              (None, 44032)        0           ['dropout_2[0][0]']              
                                                                                                  
 flatten_1 (Flatten)            (None, 11008)        0           ['dropout_5[0][0]']              
                                                                                                  
 flatten_2 (Flatten)            (None, 2752)         0           ['dropout_8[0][0]']              
                                                                                                  
 concatenate (Concatenate)      (None, 57792)        0           ['flatten[0][0]',                
                                                                  'flatten_1[0][0]',              
                                                                  'flatten_2[0][0]']              
                                                                                                  
 dropout_9 (Dropout)            (None, 57792)        0           ['concatenate[0][0]']            
                                                                                                  
 dense_6 (Dense)                (None, 10)           577930      ['dropout_9[0][0]']              
                                                                                                  
==================================================================================================
Total params: 1,791,777
Trainable params: 1,789,713
Non-trainable params: 2,064
__________________________________________________________________________________________________
Epoch 1/100
2022-04-04 21:30:10.043611: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8204
201/201 - 22s - loss: 3.3396 - accuracy: 0.3850 - top-5-accuracy: 0.8647 - val_loss: 2.1208 - val_accuracy: 0.4174 - val_top-5-accuracy: 0.8846 - 22s/epoch - 109ms/step
Epoch 2/100
201/201 - 19s - loss: 1.8098 - accuracy: 0.5117 - top-5-accuracy: 0.9360 - val_loss: 1.6814 - val_accuracy: 0.5518 - val_top-5-accuracy: 0.9450 - 19s/epoch - 93ms/step
Epoch 3/100
201/201 - 18s - loss: 1.6172 - accuracy: 0.5728 - top-5-accuracy: 0.9522 - val_loss: 1.6383 - val_accuracy: 0.5950 - val_top-5-accuracy: 0.9520 - 18s/epoch - 91ms/step
Epoch 4/100
201/201 - 19s - loss: 1.5008 - accuracy: 0.6150 - top-5-accuracy: 0.9634 - val_loss: 1.5698 - val_accuracy: 0.6076 - val_top-5-accuracy: 0.9612 - 19s/epoch - 97ms/step
Epoch 5/100
201/201 - 19s - loss: 1.4082 - accuracy: 0.6470 - top-5-accuracy: 0.9695 - val_loss: 1.3769 - val_accuracy: 0.6686 - val_top-5-accuracy: 0.9752 - 19s/epoch - 96ms/step
Epoch 6/100
201/201 - 19s - loss: 1.3250 - accuracy: 0.6808 - top-5-accuracy: 0.9746 - val_loss: 1.2986 - val_accuracy: 0.7000 - val_top-5-accuracy: 0.9728 - 19s/epoch - 97ms/step
Epoch 7/100
201/201 - 19s - loss: 1.2577 - accuracy: 0.6988 - top-5-accuracy: 0.9764 - val_loss: 1.2215 - val_accuracy: 0.7154 - val_top-5-accuracy: 0.9750 - 19s/epoch - 93ms/step
Epoch 8/100
201/201 - 20s - loss: 1.1917 - accuracy: 0.7253 - top-5-accuracy: 0.9803 - val_loss: 1.1654 - val_accuracy: 0.7418 - val_top-5-accuracy: 0.9826 - 20s/epoch - 98ms/step
Epoch 9/100
201/201 - 19s - loss: 1.1344 - accuracy: 0.7379 - top-5-accuracy: 0.9827 - val_loss: 1.1386 - val_accuracy: 0.7400 - val_top-5-accuracy: 0.9854 - 19s/epoch - 94ms/step
Epoch 10/100
201/201 - 20s - loss: 1.0823 - accuracy: 0.7542 - top-5-accuracy: 0.9842 - val_loss: 1.0547 - val_accuracy: 0.7598 - val_top-5-accuracy: 0.9834 - 20s/epoch - 101ms/step
Epoch 11/100
201/201 - 19s - loss: 1.0280 - accuracy: 0.7711 - top-5-accuracy: 0.9858 - val_loss: 0.9985 - val_accuracy: 0.7836 - val_top-5-accuracy: 0.9868 - 19s/epoch - 97ms/step
Epoch 12/100
201/201 - 20s - loss: 0.9890 - accuracy: 0.7779 - top-5-accuracy: 0.9873 - val_loss: 0.9102 - val_accuracy: 0.8016 - val_top-5-accuracy: 0.9840 - 20s/epoch - 97ms/step
Epoch 13/100
201/201 - 19s - loss: 0.9482 - accuracy: 0.7883 - top-5-accuracy: 0.9876 - val_loss: 0.9330 - val_accuracy: 0.7992 - val_top-5-accuracy: 0.9894 - 19s/epoch - 95ms/step
Epoch 14/100
201/201 - 19s - loss: 0.9060 - accuracy: 0.8014 - top-5-accuracy: 0.9893 - val_loss: 0.9147 - val_accuracy: 0.8006 - val_top-5-accuracy: 0.9862 - 19s/epoch - 94ms/step
Epoch 15/100
201/201 - 20s - loss: 0.8860 - accuracy: 0.8088 - top-5-accuracy: 0.9899 - val_loss: 0.8880 - val_accuracy: 0.8068 - val_top-5-accuracy: 0.9886 - 20s/epoch - 97ms/step
Epoch 16/100
201/201 - 20s - loss: 0.8515 - accuracy: 0.8168 - top-5-accuracy: 0.9902 - val_loss: 0.7886 - val_accuracy: 0.8334 - val_top-5-accuracy: 0.9920 - 20s/epoch - 97ms/step
Epoch 17/100
201/201 - 20s - loss: 0.8241 - accuracy: 0.8244 - top-5-accuracy: 0.9906 - val_loss: 0.7546 - val_accuracy: 0.8532 - val_top-5-accuracy: 0.9926 - 20s/epoch - 99ms/step
Epoch 18/100
201/201 - 19s - loss: 0.8053 - accuracy: 0.8315 - top-5-accuracy: 0.9919 - val_loss: 0.8641 - val_accuracy: 0.8194 - val_top-5-accuracy: 0.9896 - 19s/epoch - 95ms/step
Epoch 19/100
201/201 - 19s - loss: 0.7831 - accuracy: 0.8380 - top-5-accuracy: 0.9924 - val_loss: 0.7228 - val_accuracy: 0.8520 - val_top-5-accuracy: 0.9938 - 19s/epoch - 96ms/step
Epoch 20/100
201/201 - 19s - loss: 0.7537 - accuracy: 0.8422 - top-5-accuracy: 0.9931 - val_loss: 0.7825 - val_accuracy: 0.8406 - val_top-5-accuracy: 0.9926 - 19s/epoch - 94ms/step
Epoch 21/100
201/201 - 19s - loss: 0.7440 - accuracy: 0.8470 - top-5-accuracy: 0.9928 - val_loss: 0.7433 - val_accuracy: 0.8494 - val_top-5-accuracy: 0.9938 - 19s/epoch - 96ms/step
Epoch 22/100
201/201 - 20s - loss: 0.7260 - accuracy: 0.8531 - top-5-accuracy: 0.9932 - val_loss: 0.7049 - val_accuracy: 0.8598 - val_top-5-accuracy: 0.9946 - 20s/epoch - 98ms/step
Epoch 23/100
201/201 - 20s - loss: 0.7044 - accuracy: 0.8564 - top-5-accuracy: 0.9933 - val_loss: 0.7011 - val_accuracy: 0.8524 - val_top-5-accuracy: 0.9916 - 20s/epoch - 97ms/step
Epoch 24/100
201/201 - 19s - loss: 0.6955 - accuracy: 0.8598 - top-5-accuracy: 0.9944 - val_loss: 0.7128 - val_accuracy: 0.8614 - val_top-5-accuracy: 0.9944 - 19s/epoch - 95ms/step
Epoch 25/100
201/201 - 19s - loss: 0.6710 - accuracy: 0.8627 - top-5-accuracy: 0.9945 - val_loss: 0.7012 - val_accuracy: 0.8568 - val_top-5-accuracy: 0.9946 - 19s/epoch - 96ms/step
Epoch 26/100
201/201 - 19s - loss: 0.6653 - accuracy: 0.8663 - top-5-accuracy: 0.9949 - val_loss: 0.6649 - val_accuracy: 0.8644 - val_top-5-accuracy: 0.9958 - 19s/epoch - 96ms/step
Epoch 27/100
201/201 - 19s - loss: 0.6534 - accuracy: 0.8703 - top-5-accuracy: 0.9948 - val_loss: 0.6593 - val_accuracy: 0.8734 - val_top-5-accuracy: 0.9950 - 19s/epoch - 97ms/step
Epoch 28/100
201/201 - 20s - loss: 0.6254 - accuracy: 0.8744 - top-5-accuracy: 0.9954 - val_loss: 0.6504 - val_accuracy: 0.8766 - val_top-5-accuracy: 0.9960 - 20s/epoch - 100ms/step
Epoch 29/100
201/201 - 19s - loss: 0.6262 - accuracy: 0.8765 - top-5-accuracy: 0.9950 - val_loss: 0.6805 - val_accuracy: 0.8662 - val_top-5-accuracy: 0.9928 - 19s/epoch - 95ms/step
Epoch 30/100
201/201 - 19s - loss: 0.6097 - accuracy: 0.8801 - top-5-accuracy: 0.9957 - val_loss: 0.6525 - val_accuracy: 0.8732 - val_top-5-accuracy: 0.9954 - 19s/epoch - 94ms/step
Epoch 31/100
201/201 - 20s - loss: 0.5997 - accuracy: 0.8820 - top-5-accuracy: 0.9955 - val_loss: 0.5998 - val_accuracy: 0.8826 - val_top-5-accuracy: 0.9966 - 20s/epoch - 98ms/step
Epoch 32/100
201/201 - 19s - loss: 0.6009 - accuracy: 0.8829 - top-5-accuracy: 0.9956 - val_loss: 0.6646 - val_accuracy: 0.8626 - val_top-5-accuracy: 0.9942 - 19s/epoch - 93ms/step
Epoch 33/100
201/201 - 20s - loss: 0.5804 - accuracy: 0.8892 - top-5-accuracy: 0.9961 - val_loss: 0.6054 - val_accuracy: 0.8880 - val_top-5-accuracy: 0.9952 - 20s/epoch - 98ms/step
Epoch 34/100
201/201 - 19s - loss: 0.5689 - accuracy: 0.8887 - top-5-accuracy: 0.9963 - val_loss: 0.6087 - val_accuracy: 0.8814 - val_top-5-accuracy: 0.9958 - 19s/epoch - 96ms/step
Epoch 35/100
201/201 - 20s - loss: 0.5555 - accuracy: 0.8922 - top-5-accuracy: 0.9965 - val_loss: 0.5826 - val_accuracy: 0.8896 - val_top-5-accuracy: 0.9956 - 20s/epoch - 101ms/step
Epoch 36/100
201/201 - 19s - loss: 0.5522 - accuracy: 0.8947 - top-5-accuracy: 0.9964 - val_loss: 0.5692 - val_accuracy: 0.8912 - val_top-5-accuracy: 0.9964 - 19s/epoch - 96ms/step
Epoch 37/100
201/201 - 19s - loss: 0.5535 - accuracy: 0.8936 - top-5-accuracy: 0.9967 - val_loss: 0.6047 - val_accuracy: 0.8820 - val_top-5-accuracy: 0.9960 - 19s/epoch - 95ms/step
Epoch 38/100
201/201 - 19s - loss: 0.5399 - accuracy: 0.8972 - top-5-accuracy: 0.9966 - val_loss: 0.5550 - val_accuracy: 0.8940 - val_top-5-accuracy: 0.9964 - 19s/epoch - 97ms/step
Epoch 39/100
201/201 - 19s - loss: 0.5336 - accuracy: 0.9016 - top-5-accuracy: 0.9971 - val_loss: 0.6149 - val_accuracy: 0.8776 - val_top-5-accuracy: 0.9962 - 19s/epoch - 93ms/step
Epoch 40/100
201/201 - 19s - loss: 0.5373 - accuracy: 0.9007 - top-5-accuracy: 0.9974 - val_loss: 0.5766 - val_accuracy: 0.8856 - val_top-5-accuracy: 0.9974 - 19s/epoch - 95ms/step
Epoch 41/100
201/201 - 19s - loss: 0.5186 - accuracy: 0.9022 - top-5-accuracy: 0.9970 - val_loss: 0.5868 - val_accuracy: 0.8860 - val_top-5-accuracy: 0.9958 - 19s/epoch - 96ms/step
Epoch 42/100
201/201 - 19s - loss: 0.5040 - accuracy: 0.9038 - top-5-accuracy: 0.9973 - val_loss: 0.5807 - val_accuracy: 0.8894 - val_top-5-accuracy: 0.9962 - 19s/epoch - 95ms/step
Epoch 43/100
201/201 - 20s - loss: 0.5128 - accuracy: 0.9055 - top-5-accuracy: 0.9972 - val_loss: 0.5378 - val_accuracy: 0.9040 - val_top-5-accuracy: 0.9970 - 20s/epoch - 101ms/step
Epoch 44/100
201/201 - 19s - loss: 0.4915 - accuracy: 0.9093 - top-5-accuracy: 0.9976 - val_loss: 0.5407 - val_accuracy: 0.9000 - val_top-5-accuracy: 0.9970 - 19s/epoch - 94ms/step
Epoch 45/100
201/201 - 19s - loss: 0.4855 - accuracy: 0.9110 - top-5-accuracy: 0.9977 - val_loss: 0.5894 - val_accuracy: 0.8912 - val_top-5-accuracy: 0.9974 - 19s/epoch - 95ms/step
Epoch 46/100
201/201 - 19s - loss: 0.4898 - accuracy: 0.9117 - top-5-accuracy: 0.9978 - val_loss: 0.5389 - val_accuracy: 0.8962 - val_top-5-accuracy: 0.9962 - 19s/epoch - 94ms/step
Epoch 47/100
201/201 - 19s - loss: 0.4789 - accuracy: 0.9104 - top-5-accuracy: 0.9975 - val_loss: 0.5786 - val_accuracy: 0.8896 - val_top-5-accuracy: 0.9960 - 19s/epoch - 94ms/step
Epoch 48/100
201/201 - 19s - loss: 0.4772 - accuracy: 0.9139 - top-5-accuracy: 0.9978 - val_loss: 0.5592 - val_accuracy: 0.8940 - val_top-5-accuracy: 0.9968 - 19s/epoch - 94ms/step
Epoch 49/100
201/201 - 19s - loss: 0.4783 - accuracy: 0.9140 - top-5-accuracy: 0.9978 - val_loss: 0.5183 - val_accuracy: 0.9030 - val_top-5-accuracy: 0.9972 - 19s/epoch - 94ms/step
Epoch 50/100
201/201 - 19s - loss: 0.4639 - accuracy: 0.9166 - top-5-accuracy: 0.9979 - val_loss: 0.4967 - val_accuracy: 0.9022 - val_top-5-accuracy: 0.9968 - 19s/epoch - 93ms/step
Epoch 51/100
201/201 - 19s - loss: 0.4314 - accuracy: 0.9220 - top-5-accuracy: 0.9981 - val_loss: 0.5170 - val_accuracy: 0.8966 - val_top-5-accuracy: 0.9952 - 19s/epoch - 94ms/step
Epoch 52/100
201/201 - 19s - loss: 0.4188 - accuracy: 0.9238 - top-5-accuracy: 0.9984 - val_loss: 0.5295 - val_accuracy: 0.8990 - val_top-5-accuracy: 0.9956 - 19s/epoch - 94ms/step
Epoch 53/100
201/201 - 19s - loss: 0.4292 - accuracy: 0.9223 - top-5-accuracy: 0.9982 - val_loss: 0.5367 - val_accuracy: 0.8970 - val_top-5-accuracy: 0.9968 - 19s/epoch - 96ms/step
Epoch 54/100
201/201 - 19s - loss: 0.4242 - accuracy: 0.9235 - top-5-accuracy: 0.9983 - val_loss: 0.5083 - val_accuracy: 0.9048 - val_top-5-accuracy: 0.9972 - 19s/epoch - 94ms/step
Epoch 55/100
201/201 - 19s - loss: 0.4285 - accuracy: 0.9224 - top-5-accuracy: 0.9986 - val_loss: 0.5075 - val_accuracy: 0.9010 - val_top-5-accuracy: 0.9964 - 19s/epoch - 95ms/step
Train- 1 : The total training Time is :  1061.9597852230072  seconds
313/313 - 2s - loss: 0.5242 - accuracy: 0.9011 - top-5-accuracy: 0.9968 - 2s/epoch - 5ms/step
Train-1: Test accuracy: 90.11%
Train-1: Test top 5 accuracy: 99.68%
Train- 1 : The total testing Time is :  1.8438770771026611  seconds
```