from random import random
import torch
import numpy as np
import onnxruntime
from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader
from utils.ClsDataset import Inference_Dataset
import os
from PIL import Image
import cv2

def inference(image_folder_path:str, model_path:str, tagName:dict):
    # cudaDevice = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
    print('onnxruntime:' ,onnxruntime.get_device())

    # 小圖用的transform
    dataTransform_small = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    testSet = Inference_Dataset(Path(image_folder_path))
    dataLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False, num_workers=4)
    
    ### Model define ###
    session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    answer = []
    answerName = []

    print('Step(1/4) start inference')
    print(f'input data: {len(dataLoader)}')
    countOK = 0
    countNG = 0
    for i, data in enumerate(dataLoader):
        # tensor轉回numpy後再把維度轉回來且乘255
        inputs = data[0].cpu().numpy().transpose(2, 3, 1, 0)[:,:,:,-1] * 255

        draw_input = inputs.copy()
        draw_input = cv2.cvtColor(np.asarray(inputs), cv2.COLOR_RGB2BGR)
        totalcount = 0
        
        print(f'    {i+1} / {len(testSet)}')
        print('Name: ', testSet.filename[i])

        try:
            # crop txt
            with open(f'{os.path.join("./utils/gseo_preprocessing/", testSet.filename[i][-10:-4])}.txt', 'r') as txtFile:
                txtList = txtFile.readlines()
        except:
            raise BaseException(f"make sure your image name({testSet.filename[i]}) is correct !!!")

        patchOK = 0
        patchNG = 0
        heavy_defect = 0

        for txtLine in txtList:
            txtIndex = txtLine.replace('\n', '').split(' ')
            imageCrop = inputs[int(txtIndex[2]):int(txtIndex[4]), int(txtIndex[1]):int(txtIndex[3])]
            # numpy to PIL
            PIL_image = Image.fromarray(np.uint8(imageCrop)).convert('RGB')
            imageCrop = dataTransform_small(PIL_image)
            # 把batch的維度補回來
            imageCrop = imageCrop.unsqueeze(0) 

            totalcount+=1

            result = session.run([], {"input": imageCrop.cpu().numpy()})

            output1 = torch.tensor(result)
            output1 = output1[0,:,:]
            outputsSoftmax = torch.nn.functional.softmax(output1, dim=1)
            confidenceScore, predicted = torch.max(outputsSoftmax, 1)
            predicted = int(predicted)

            if int(predicted) == 1:
                patchOK += 1
            else:
               # 是否為嚴重瑕疵       
                if float(confidenceScore) >= 0.9:
                    # print('Found heavy defect: ',confidenceScore, predicted)
                    # 畫框
                    draw_input = cv2.rectangle(draw_input, (int(txtIndex[1]), int(txtIndex[2])), (int(txtIndex[3]), int(txtIndex[4])), (255,255,0), 5)
                    heavy_defect = 1
                else:
                    draw_input = cv2.rectangle(draw_input, (int(txtIndex[1]), int(txtIndex[2])), (int(txtIndex[3]), int(txtIndex[4])), (0,255,0), 5)
                patchNG += 1
        cv2.imwrite(f'./output/{testSet.filename[i]}', draw_input)
        # 小圖推論完成後
        print(f'patch OK: {patchOK}/{totalcount}')
        print(f'patch NG: {patchNG}/{totalcount}')
        # 筏值為10 可修改。
        if patchNG >= 10 or heavy_defect == 1:
            ans = 'NG'
            countNG+=1
            answer.append(str(0)) # predicted = [chinese, english] or unknown
            answerName.append(tagName[str(0)])
        else:
            ans = 'OK'
            countOK+=1
            answer.append(str(1))
            answerName.append(tagName[str(1)])
        print('predict: ', ans)
        
    print(f'Total OK datas: {countOK}/{len(dataLoader)}')
    print(f'Total NG datas:: {countNG}/{len(dataLoader)}')
    finalResult = {'FileName': testSet.filename, 'Prediction': answer, 'Prediction_Name':answerName}
    print('Step(2/4) inference finished')
    return finalResult