import os
import time
from utils import ModelInference
from utils.UsingCsv import UsingCsv
from configparser import ConfigParser


def inference(image_folder_path:str, model_path:str, MAMC_path:str):

    fileName = os.path.split(model_path)[-1]
    config = ConfigParser()
    config.read(model_path.replace('.onnx', '.ini'), encoding='utf-8')
    labels = config['CLASSES']
    postProcessing = config['POSTPROCESSING']
    classNumber = len(labels.items())
    tagName = {**labels, **postProcessing}

    if "GSEO" in fileName:
        finalResult = ModelInference.inference(image_folder_path, model_path, tagName)   
    else:
        raise BaseException("The onnx file name in model_path must include 'AoiColorModel' or 'AoiGrayModel' in capital letter.")

    print('Step(3/4) csv start writing')
    fileName = f'{fileName.split(".")[0]}.csv'
    csvFile = UsingCsv(fileName,'.')
    csvFile.create_csv()
    csvFile.writing(['NAS_path', image_folder_path],'a')
    csvFile.writing(['AI model', model_path],'a')
    csvFile.writing(['file_name', 'class', 'prediction'],'a')

    for i in range(len(finalResult['FileName'])):
        if finalResult['Prediction'][i] == str(classNumber):
            csvFile.writing([finalResult['FileName'][i], 0, finalResult['Prediction'][i]],'a')
        else:
            csvFile.writing([finalResult['FileName'][i], 
                            int(finalResult['Prediction'][i]), 
                            finalResult['Prediction_Name'][i]], 'a')
    
    nowTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    saveName = f'{MAMC_path}/{nowTime}_result.csv'

    if os.path.exists(f'{saveName}'):
        os.remove(f'{saveName}')

    os.rename(fileName, f'{saveName}')
    print('Step(4/4) csv has been written')
    return finalResult