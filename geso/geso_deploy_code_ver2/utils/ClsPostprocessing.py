import torch
from torch import tensor

def confidence_threshold(output, className, selectLabel, confidenTh=0.75):
    '''
        過濾指定類別(selectLabel)希望信心分數高於門檻(confidenTh)，否則選擇次高的結果，輸出最高分數的結果

        Args:
            output: 最後輸出結果
            className: 按照模型順序的類別名稱
            selectLabel: 選擇要卡分數的類別
            confidenTh: 信心分數門檻值
            
        Return:
            newOutput: 過濾過的新輸出結果
    '''
    output = torch.tensor(output)
    output = output[0,:,:]
    outputsSoftmax = torch.nn.functional.softmax(output, dim=1)
    confidenceScore, predicted = torch.max(outputsSoftmax, 1)
    confidenLabelNumber = None
    for i in range(len(className)):
        if className[i] == selectLabel:
            confidenLabelNumber = i   # 紀錄指定類別的位置
    # assert isinstance(confidenLabelNumber, int), f'you selected label: "{selectLabel}" does not exit, model label: {className}'
    
    if predicted[0] == confidenLabelNumber and confidenceScore[0] < confidenTh:  
        # print(f'{className[confidenLabelNumber]}\'s sorce is {confidenceScore[0]}, and it has been filtered out')
        output[0][confidenLabelNumber] = 0  #把原本最高分數改成0

    return output
    

def unknown_threshold(unknownThreshold, output, classNumber, reverse=False):
    """
    Filter out images with scores below the threshold 

    Args:
        unknownThreshold (float): confidence threshold of unknown
        output (list): model prediction
        reverse (boolean): the threshold arrange
    """

    filerDict = {"Unknown": unknownThreshold}
    filer = sorted(filerDict.items(), key=lambda x:x[1], reverse=reverse)

    output = torch.tensor(output)
    output = output[0,:,:]
    outputsSoftmax = torch.nn.functional.softmax(output, dim=1)
    confidenceScore, predicted = torch.max(outputsSoftmax, 1)
    for tagName, threshold in filer:
        if confidenceScore[0] < threshold and not reverse:
            return int(classNumber), outputsSoftmax
        elif confidenceScore[0] > threshold and reverse:
            return int(classNumber), outputsSoftmax
    return int(predicted), outputsSoftmax
    