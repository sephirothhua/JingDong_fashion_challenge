
import numpy as np

class config():
    Name = "ImageNet"

    Base_model_name = "Densenet121"

    Pooling_type = None

    Input_shape = (224,224,3)

    Step_per_epoch = 1000

    Validation_steps = 50

    Num_categrade = 13

    Learning_rate = 1e-2

    Use_learning_rate_reduce = False

    Learning_rate_decay = 0.5e-6

    Batch_size = 1

    Val_batch_size = 64

    Mean_pixel = np.array([170.8, 162.6, 159.5])

    Sampledata_distributes = np.array([104,4292,49893,262,18722,8882,197,1356,49389,7094,3712,1249,361])



