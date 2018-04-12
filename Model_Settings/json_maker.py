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
baseTrainDataDir = '../Data/raw_labeled/tfrecords/'
baseTestDataDir = '../Data/raw_labeled/tfrecords/'
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
        'initialLearningRate' : 0.005,
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

    if runName == '180410_clsf_smce': # using 171003_ITR_B but with gaussian location softmax loss for all last tuple
        dataLocal['classificationModel'] = True
        _180410_clsf_smce(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    ####
    elif runName == '180410_clsf_g_smce': # using 171003_ITR_B but with gaussian location softmax loss for all last tuple
        dataLocal['classificationModel'] = True
        _180410_clsf_g_smce(reCompile, trainLogDirBase, testLogDirBase, runName, dataLocal)
    else:
        print("--error: Model name not found!")
        return False
    return True
    ##############
    ##############
    ##############

def _180410_clsf_smce(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_8l2f_inception'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 512]
        data['trainBatchSize'] = 8#8#16
        data['testBatchSize'] = 8#8#16
        data['numTrainDatasetExamples'] = 17311
        data['numTestDatasetExamples'] = 4327
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"
        
        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['testDataDir'] = '../Data/CBY/test_tfrecs/' ############ this should be updated
        data['trainDataDir'] = '../Data/raw_labeled/tfrecords/'
        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

def _180410_clsf_g_smce(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_8l2f_inception'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 512]
        data['trainBatchSize'] = 8#8#16
        data['testBatchSize'] = 8#8#16
        data['numTrainDatasetExamples'] = 17311
        data['numTestDatasetExamples'] = 4327
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_gaussian_softmaxCrossentropy_loss"
        
        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        
        data['trainDataDir'] = '../Data/raw_labeled/tfrecords/'
        data['testDataDir'] = '../Data/CBY/test_tfrecs/' ############ this should be updated
        data['trainOutputDir'] = data['trainLogDir']+'/target/'
        data['testOutputDir'] = data['testLogDir']+'/target/'
        _set_folders(data['trainOutputDir'])
        _set_folders(data['testOutputDir'])
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)


def _180412_clsf_smce(reCompile, trainLogDirBase, testLogDirBase, runName, data):
    if reCompile:
        data['modelName'] = 'cnn_8l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['trainBatchSize'] = 8#8#16
        data['testBatchSize'] = 8#8#16
        data['numTrainDatasetExamples'] = 17311
        data['numTestDatasetExamples'] = 4327
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_softmaxCrossentropy_loss"
        
        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName

        data['testDataDir'] = '../Data/CBY/test_tfrecs/' ############ this should be updated
        data['trainDataDir'] = '../Data/raw_labeled/tfrecords/'
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
        data['numTrainDatasetExamples'] = 17311
        data['numTestDatasetExamples'] = 4327
        data['logicalOutputSize'] = 6
        data['networkOutputSize'] = data['logicalOutputSize']
        data['lossFunction'] = "_params_classification_gaussian_softmaxCrossentropy_loss"
        
        ## runs
        data['trainMaxSteps'] = 75000
        data['numEpochsPerDecay'] = float(data['trainMaxSteps']/3)

        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        
        data['trainDataDir'] = '../Data/raw_labeled/tfrecords/'
        data['testDataDir'] = '../Data/CBY/test_tfrecs/' ############ this should be updated
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
