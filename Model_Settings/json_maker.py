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
        'numTrainDatasetExamples' : 17311,
        'numTestDatasetExamples' : 4327,
        'trainDataDir' : baseTrainDataDir,
        'testDataDir' : baseTestDataDir,
        'trainLogDir' : trainLogDirBase+'',
        'testLogDir' : testLogDirBase+'',
        'outputTrainDir' : trainLogDirBase+'/target/',
        'outputTestDir' : testLogDirBase+'/target/',
        'pretrainedModelCheckpointPath' : '',
        # Image Parameters
        'pngRows' : 480,
        'pngCols' : 640,
        'pngChannels' : 1, # All PCL files should have same cols
        # Model Parameters
        'modelName' : '',
        'modelShape' : [64, 64, 64, 64, 128, 128, 128, 128, 1024],
        'batchNorm' : True,
        'weightNorm' : False,
        'optimizer' : 'MomentumOptimizer', # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        'momentum' : 0.9,
        'initialLearningRate' : 0.01,
        'learningRateDecayFactor' : 0.1,
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
    elif runName == '180412_clsf_g_smce': # using 171003_ITR_B but with gaussian location softmax loss for all last tuple
        dataLocal['classificationModel'] = True
        _180412_clsf_g_smce(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180418_clsf_smce': # binary classification  >=2 or not
        dataLocal['classificationModel'] = True
        _180418_clsf_smce(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180809': # using 171003_ITR_B but with gaussian location softmax loss for all last tuple
        dataLocal['classificationModel'] = True
        _180809(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180810': # using 171003_ITR_B but with gaussian location softmax loss for all last tuple
        dataLocal['classificationModel'] = True
        _180810(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180814c2': # using 171003_ITR_B but with gaussian location softmax loss for all last tuple
        dataLocal['classificationModel'] = True
        _180814c2(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    else:
        print("--error: Model name not found!")
        return False
    return True
    ##############
    ##############
    ##############

def _180412_clsf_smce(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_8l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['trainBatchSize'] = 8#8#16
        data['testBatchSize'] = 8#8#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"
        
        data['pngRows'] = 240
        data['pngCols'] = 320

        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/raw_labeled/train_tfrecs/'
        data['testDataDir'] = '../Data/raw_labeled/test_tfrecs/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _180412_clsf_g_smce(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_8l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['trainBatchSize'] = 8#8#16
        data['testBatchSize'] = 8#8#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_gaussian_softmaxCrossentropy_loss"
        
        data['pngRows'] = 240
        data['pngCols'] = 320
        
        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        
        data['trainDataDir'] = '../Data/raw_labeled/train_tfrecs/'
        data['testDataDir'] = '../Data/raw_labeled/test_tfrecs/' 

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _180418_clsf_smce(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_8l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['trainBatchSize'] = 8#8#16
        data['testBatchSize'] = 8#8#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 2
        data['outputSize']=2
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"
        
        data['pngRows'] = 240
        data['pngCols'] = 320

        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/raw_labeled/train_tfrecs_b_2/'
        data['testDataDir'] = '../Data/raw_labeled/test_tfrecs_b_2/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _180809(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_4l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [64, 96, 96, 128, 512]
        data['trainBatchSize'] = 8#16
        data['testBatchSize'] = 8#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"
        
        data['pngRows'] = 240
        data['pngCols'] = 320
        
        ## runs
        data['trainMaxSteps'] = 120000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        
        data['trainDataDir'] = '../Data/raw_labeled/train_tfrecs/'
        data['testDataDir'] = '../Data/raw_labeled/test_tfrecs/' 

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
    
def _180810(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_4l2f_new'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        #data['modelShape'] = [48, 96, 96, 128, 512]
        data['modelShape'] = [16, 32, 0, 0, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 32#16
        data['numTrainDatasetExamples'] = 21020
        data['numTestDatasetExamples'] = 131
        data['logicalOutputSize'] = 2
        data['outputSize']=2
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"
        
        data['pngRows'] = 240
        data['pngCols'] = 320

        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)
        data['testMaxSteps'] = int(data['numTestDatasetExamples']/data['testBatchSize'])+1

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['trainDataDir'] = '../Data/raw_labeled/train_tfrecs_b_2/'
        data['testDataDir'] = '../Data/raw_labeled/test_tfrecs_b_2/'

        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _180814c2(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_4l2f_new'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        #data['modelShape'] = [48, 96, 96, 128, 512]
        data['modelShape'] = [16, 32, 0, 0, 256]
        data['trainBatchSize'] = 32#16
        data['testBatchSize'] = 32#16
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
####################################################################################
####################################################################################
####################################################################################

def recompile_json_files(runName):
    success = write(runName)
    if success:
        print("JSON files updated")
    return success
