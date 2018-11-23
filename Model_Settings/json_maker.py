import json
import collections
import numpy as np
import os

def write_json_file(filename, datafile):
    filename = 'Model_Settings/'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent=0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

####################################################################################
####################################################################################
####################################################################################

# REGRESSION --- Twin Common Parameters
#baseTrainDataDir = '../Data/kitti/train_tfrecords'
#baseTestDataDir = '../Data/kitti/test_tfrecords'
#trainLogDirBase = '../Data/kitti/logs/tfdh_twin_py_logs/train_logs/'
#testLogDirBase = '../Data/kitti/logs/tfdh_twin_py_logs/test_logs/'

# CLASSIFICATION --- Twin Correlation Matching Common Parameters
baseTrainDataDir = '../Data/raw_labeled/train_tfrecs/'
baseTestDataDir = '../Data/raw_labeled/test_tfrecs/'
trainLogDirBase = '../Data/logs/clsf_logs/train_logs/'
testLogDirBase = '../Data/logs/clsf_logs/test_logs/'



####################################################################################
####################################################################################
####################################################################################
reCompileJSON = True
####################################################################################
####################################################################################
####################################################################################

def write(runName):
    dataLocal = {
        # Data Parameters
        'numTrainDatasetExamples' : 23640,
        'numTestDatasetExamples' : 131,
        'trainDataDir' : baseTrainDataDir,
        'testDataDir' : baseTestDataDir,
        'trainLogDir' : trainLogDirBase+'',
        'testLogDir' : testLogDirBase+'',
        'outputTrainDir' : trainLogDirBase+'/target/',
        'outputTestDir' : testLogDirBase+'/target/',
        'pretrainedModelCheckpointPath' : '',
        # Image Parameters
        'pngRows' : 256,
        'pngCols' : 352,
        'pngChannels' : 1, # All PCL files should have same cols
        # Model Parameters
        'modelName' : '',
        'modelShape' : [64, 64, 64, 64, 128, 128, 128, 128, 1024],
        'batchNorm' : True,
        'weightNorm' : False,
        'optimizer' : 'MomentumOptimizer', # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        'momentum' : 0.9,
        'initialLearningRate' : 0.01,
        'learningRateDecayFactor' : 0.01,
        'numEpochsPerDecay' : 10000.0,
        'epsilon' : 0.1,
        'dropOutKeepRate' : 0.5,
        'clipNorm' : 1.0,
        'lossFunction' : 'L2',
        # Train Parameters
        'trainBatchSize' : 16,
        'testBatchSize' : 16,
        'outputSize' : 6, # 6 labels
        'trainMaxSteps' : 90000,
        'testMaxSteps' : 1,
        'usefp16' : False,
        'logDevicePlacement' : False,
        'classification' : False,
        }
    dataLocal['testMaxSteps'] = int(np.ceil(dataLocal['numTestDatasetExamples']/dataLocal['testBatchSize']))

    reCompile = True
    NOreCompile = False

    if runName == '180412_clsf_smce': # using 171003_ITR_B but with gaussian location softmax loss for all last tuple
        dataLocal['classificationModel'] = True
        _180412_clsf_smce(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180911c2': # using 180814c2 inceptionized 
        dataLocal['classificationModel'] = True
        _180911c2(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180912c2': # using 180814c2 inceptionized 
        dataLocal['classificationModel'] = True
        _180912c2(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180913c2': # using 180814c2 inceptionized 
        dataLocal['classificationModel'] = True
        _180913c2(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180916c2': # using 180814c2 inceptionized 
        dataLocal['classificationModel'] = True
        _180916c2(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180912c2new': # using 180814c2 
        dataLocal['classificationModel'] = True
        _180912c2new(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180912c2new2': # using 180814c2 
        dataLocal['classificationModel'] = True
        _180912c2new2(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '181113c2': # using 1800912c2new2 
        dataLocal['classificationModel'] = True
        _181113c2(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '181113c1': # using 1800912c2new2 
        dataLocal['classificationModel'] = True
        _181113c1(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '181114c1': # using 1800912c2new2 
        dataLocal['classificationModel'] = True
        _181114c1(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '181114c2': # using 1800912c2new2 
        dataLocal['classificationModel'] = True
        _181114c2(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '181114rg0': # using 1800912c2new2 
        dataLocal['classificationModel'] = True
        _181114rg0(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '181114rgm': # using 1800912c2new2 
        dataLocal['classificationModel'] = True
        _181114rgm(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '181121c2': # using 1800912c2new2 
        dataLocal['classificationModel'] = True
        _181121c2(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '181121rgm': # using 1800912c2new2 
        dataLocal['classificationModel'] = True
        _181121rgm(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    else:
        print("--error: Model name not found!")
        return False
    return True
    ##############
    ##############
    ##############
def _180911c2(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_8l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        #data['modelShape'] = [48, 96, 96, 128, 512]
        data['modelShape'] = [64, 64, 80, 80, 96, 96, 128, 128, 256]
        data['trainBatchSize'] = 16#32#16
        data['testBatchSize'] = 16#32#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 240
        data['pngCols'] = 320
        data['pngChannels'] = 2
        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_2c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_2c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_2c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
def _180912c2(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_8l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        #data['modelShape'] = [48, 96, 96, 128, 512]
        data['modelShape'] = [64, 64, 80, 80, 96, 96, 128, 128, 128, 128]
        data['trainBatchSize'] = 16#32#16
        data['testBatchSize'] = 16#32#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 240
        data['pngCols'] = 320
        data['pngChannels'] = 2
        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_2c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_2c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_2c/'


        #data['dropOutKeepRate'] = 0.7
        #data['optimizer'] = 'AdamOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        #data['momentum'] = 0.9
        #data['initialLearningRate'] = 0.01
        #data['learningRateDecayFactor'] = 0.01
        #data['epsilon'] = 0.1


        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
def _180912c2new(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_8l2f_new'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        #data['modelShape'] = [0   1   2   3   4   5    6    7    8    9]
        data['modelShape'] = [64, 64, 80, 80, 96, 96, 128, 128, 256, 256]
        data['trainBatchSize'] = 32#32#16
        data['testBatchSize'] = 32#32#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "focal_loss"#"_params_classification_softmaxCrossentropy_loss"
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 2
        ## runs
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.01
        data['learningRateDecayFactor'] = 0.01
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_2c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_2c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_2c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = False
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _180912c2new2(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_8l2f_new'
        
        data['dropOutKeepRate'] = 0.3
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.01
        data['learningRateDecayFactor'] = 0.1
        data['epsilon'] = 0.1
        
        #data['modelShape'] = [0   1   2   3   4   5    6    7    8    9]
        data['modelShape'] = [64, 64, 80, 80, 96, 96, 128, 128, 256, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 32#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "focal_loss"#"ohem_loss"#"_params_classification_softmaxCrossentropy_loss"#"focal_loss"#
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 2
        ## runs
        data['trainMaxSteps'] = 45000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_2c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_2c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_2c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)


def _180913c2(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_10l'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [64, 64, 80, 80, 96, 96, 128, 128, 64]
        data['trainBatchSize'] = 16#16
        data['testBatchSize'] = 16#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 240
        data['pngCols'] = 320
        data['pngChannels'] = 2
        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_2c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_2c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_2c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _180916c2(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_11l'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [64, 64, 80, 80, 96, 96, 128, 128, 64]
        data['trainBatchSize'] = 16#16
        data['testBatchSize'] = 16#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 240
        data['pngCols'] = 320
        data['pngChannels'] = 2
        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_2c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_2c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_2c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)


def _181113c2_91_6(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_4lf'
        
        data['dropOutKeepRate'] = 0.5
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.0005
        data['learningRateDecayFactor'] = 0.1
        data['epsilon'] = 0.1
        
        #data['modelShape'] = [0   1    2    3    4]
        data['modelShape'] = [64, 128, 256, 512, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 1#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "clsf_smce_l2reg" # 88.549 - 96.183 @ 8250
        #data['lossFunction'] = "clsf_ohem_l2reg" # 91.603 - 98.473 @ 4250

        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 2
        ## runs
        data['trainMaxSteps'] = 20010
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_2c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_2c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_2c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _181113c2(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_4lf'
        
        data['dropOutKeepRate'] = 0.5
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.001
        data['learningRateDecayFactor'] = 0.1
        data['epsilon'] = 0.1
        
        #data['modelShape'] = [0   1    2    3    4]
        data['modelShape'] = [64, 128, 256, 512, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 1#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "clsf_smce_l2reg"#"ohem_loss"#"_params_classification_softmaxCrossentropy_loss"#"focal_loss"#clsf_smce_l2reg#clsf_ohem_l2reg
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 2
        ## runs
        data['trainMaxSteps'] = 20010
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_2c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_2c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_2c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _181113c1(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_4lf'
        
        data['dropOutKeepRate'] = 0.5
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.001
        data['learningRateDecayFactor'] = 0.1
        data['epsilon'] = 0.1
        
        #data['modelShape'] = [0   1    2    3    4]
        data['modelShape'] = [64, 128, 256, 512, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 1#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "clsf_ohem_l2reg"#"ohem_loss"#"_params_classification_softmaxCrossentropy_loss"#"focal_loss"#clsf_smce_l2reg#clsf_ohem_l2reg
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 1
        ## runs
        data['trainMaxSteps'] = 20010
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_1c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_1c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_1c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _181114c1(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_6lf'
        
        data['dropOutKeepRate'] = 0.5
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.0005
        data['learningRateDecayFactor'] = 0.1
        data['epsilon'] = 0.1
        
        #data['modelShape'] = [0   1    2    3    4]
        data['modelShape'] = [64, 128, 256, 512, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 1#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "clsf_smce_l2reg"#"ohem_loss"#"_params_classification_softmaxCrossentropy_loss"#"focal_loss"#clsf_smce_l2reg#clsf_ohem_l2reg
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 1
        ## runs
        data['trainMaxSteps'] = 20010
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_1c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_1c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_1c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _181114c2(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_6lf'
        
        data['dropOutKeepRate'] = 0.5
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.0005
        data['learningRateDecayFactor'] = 0.1
        data['epsilon'] = 0.1
        
        #data['modelShape'] = [0   1    2    3    4]
        data['modelShape'] = [64, 128, 256, 512, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 1#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "clsf_smce_l2reg"#"ohem_loss"#"_params_classification_softmaxCrossentropy_loss"#"focal_loss"#clsf_smce_l2reg#clsf_ohem_l2reg
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 2
        ## runs
        data['trainMaxSteps'] = 20010
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_2c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_2c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_2c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _181121c2(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'mobilenet'
        
        #data['dropOutKeepRate'] = 0.5
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.0005
        data['learningRateDecayFactor'] = 0.1
        data['epsilon'] = 0.1
        
        #data['modelShape'] = [0   1    2    3    4]
        #data['modelShape'] = [64, 128, 256, 512, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 1#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"#"ohem_loss"#"_params_classification_softmaxCrossentropy_loss"#"focal_loss"#clsf_smce_l2reg#clsf_ohem_l2reg
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 2
        ## runs
        data['trainMaxSteps'] = 20010
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_2c/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_2c/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_2c/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)


def _181121rgm(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'mobilenet'
        
        #data['dropOutKeepRate'] = 0.5
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.0005
        data['learningRateDecayFactor'] = 0.1
        data['epsilon'] = 0.1
        
        #data['modelShape'] = [0   1    2    3    4]
        #data['modelShape'] = [64, 128, 256, 512, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 1#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"#"ohem_loss"#"_params_classification_softmaxCrossentropy_loss"#"focal_loss"#clsf_smce_l2reg#clsf_ohem_l2reg
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 3
        ## runs
        data['trainMaxSteps'] = 20010
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_rgm_1/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_rgm_1/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_rgm_1/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)


def _181114rg0(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_6lf'
        
        data['dropOutKeepRate'] = 0.5
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.0005
        data['learningRateDecayFactor'] = 0.1
        data['epsilon'] = 0.1
        
        #data['modelShape'] = [0   1    2    3    4]
        data['modelShape'] = [64, 128, 256, 512, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 1#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "clsf_smce_l2reg"#"ohem_loss"#"_params_classification_softmaxCrossentropy_loss"#"focal_loss"#clsf_smce_l2reg#clsf_ohem_l2reg
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 3
        ## runs
        data['trainMaxSteps'] = 20010
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_rg0_1/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_rg0_1/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_rg0_1/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)


def _181114rgm(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_6lf'
        
        data['dropOutKeepRate'] = 0.5
        data['optimizer'] = 'AdaGrad' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['momentum'] = 0.9
        data['initialLearningRate'] = 0.0005
        data['learningRateDecayFactor'] = 0.1
        data['epsilon'] = 0.1
        
        #data['modelShape'] = [0   1    2    3    4]
        data['modelShape'] = [64, 128, 256, 512, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 1#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['outputSize']=6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "clsf_smce_l2reg"#"ohem_loss"#"_params_classification_softmaxCrossentropy_loss"#"focal_loss"#clsf_smce_l2reg#clsf_ohem_l2reg
        
        ######## No resizing - images are resized after parsing inside data_input.py
        data['pngRows'] = 256
        data['pngCols'] = 352
        data['pngChannels'] = 3
        ## runs
        data['trainMaxSteps'] = 20010
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1
        
        data['numValiDatasetExamples'] = 1024
        data['valiSteps'] = int(data['numValiDatasetExamples']/data['trainBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/cold_wb/train_tfrecs_rgm_1/'
        data['valiDataDir'] = '../Data/cold_wb/vali_tfrecs_rgm_1/'
        data['testDataDir'] = '../Data/cold_wb/test_tfrecs_rgm_1/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

####################################################################################
####################################################################################
####################################################################################

def recompile_json_files(runName):
    success = write(runName)
    if success:
        print("JSON files updated")
    return success
