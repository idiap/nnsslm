from .base import (ACT_NONE, ACT_SIGMOID, ACT_TANH, ACT_RELU, ACT_INSTRUCTION,
                   ACT_SIG_NOSAT, ACT_SIG_10, load_module, num_params,
                   POOL_NONE, POOL_MAX, POOL_AVG, POOL_STAT, POOL_WEIGHTED_AVG,
                   POOL_WEIGHTED_STAT, is_module_gpu, WeightInitializer,
                   CircularPadding2d)
from .train import (train_nn, train_stage1, stage1_loss, cross_entropy_loss,
                    cross_entropy_loss_2, CrossEntropyLossOnSM, Stage1Loss,
                    adapt_nn, adapt_decomposed)
from .archs import MLP, CNN, RegionFC
from .multitask import ResNetTwoStage, SslSnscLoss, AddConstantSns, \
                       ResNetDomainClassifier
from .multitask_v2 import DoaMultiTaskResnet, DoaSingleTaskResnet, DoaResnetTrunk
from .multitask_iterative import DoaMultiTaskIterative
from .multitask_ms import DoaConvHourglass, DoaMultiTaskMultiStage
from .dann import train_dann
from .triplet import train_multitask_triplet, TripletLoss, LossExpandToTriplet, \
                     MultitaskLossOnTriplet
from .testing import ResNetTwoStageConfig, ResNetTwoStageCustomized, \
                     FullyConvMaxPoolOut
from .obsolete import ResNet, ResNetv2, ResNetCtx32, MLP_Softmax, ResNetClassification
from .gradcam import compute_gradcam, GradCamable
from .mt_unidoa import UniformDoaFeatureNet
