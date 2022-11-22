from utils import MAMC_model

if __name__ == '__main__':

    finalResult = MAMC_model.inference(image_folder_path = './input',
                                       model_path = './model/GSEO_ver2_0.onnx',
                                       MAMC_path = './result')

    for i in range(len(finalResult['FileName'])):
        print("FileName: {:20s} / Prediction: {:2s} / Prediction Name: {:15s}".format(finalResult['FileName'][i], finalResult['Prediction'][i], finalResult['Prediction_Name'][i]))