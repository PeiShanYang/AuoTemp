import cv2, json, os

with open('geso.json', 'r') as jsonFile:
    jsonData = json.load(jsonFile)

imageFolderPath = jsonData["imageFolderPath"]
fileType = jsonData["fileType"]
fileList = [_.rstrip(fileType) for _ in os.listdir(imageFolderPath) if _.endswith(fileType)]
for fileName in fileList:
    image = cv2.imread(f'{os.path.join(imageFolderPath, fileName)}{fileType}')
    with open(f'{os.path.join(imageFolderPath, fileName)}.txt', 'r') as txtFile:
        txtList = txtFile.readlines()
    for txtLine in txtList:
        txtIndex = txtLine.replace('\n', '').split(' ')
        imageCrop = image[int(txtIndex[2]):int(txtIndex[4]), int(txtIndex[1]):int(txtIndex[3])]
        cv2.imwrite(f'{os.path.join(imageFolderPath, fileName)}_{txtIndex[1]}_{txtIndex[2]}{fileType}', imageCrop) 

