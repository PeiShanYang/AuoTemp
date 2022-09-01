import cv2, json, os

imageFolderPath = 'NG(BMP)'

fileList = [_.rstrip('.BMP') for _ in os.listdir(imageFolderPath) if _.endswith('.BMP')]

map = cv2.imread(f'{os.path.join(imageFolderPath, fileList[0])}.BMP')

for fileName in fileList:
    image = cv2.imread(f'{os.path.join(imageFolderPath, fileName)}.BMP')
    with open(f'{os.path.join(imageFolderPath, fileName)}.txt', 'r') as txtFile:
        txtList = txtFile.readlines()
    for txtLine in txtList:
        txtIndex = txtLine.replace('\n', '').split(' ')
        cv2.rectangle(image, (int(txtIndex[1]), int(txtIndex[2])), (int(txtIndex[3]), int(txtIndex[4])), (0, 0, 255), 4)
        cv2.rectangle(map, (int(txtIndex[1]), int(txtIndex[2])), (int(txtIndex[3]), int(txtIndex[4])), (0, 0, 255), -1)
    image = cv2.resize(image, (1000, 1000))
    cv2.imwrite(f'{os.path.join(imageFolderPath, fileName)}_preview.BMP', image)
    # imageShow = cv2.resize(image, (1200, 1200))
    # cv2.imshow('image', imageShow)
    # cv2.waitKey(0)
map = cv2.resize(map, (1000, 1000))
cv2.imwrite(f'{imageFolderPath}/map_preview.BMP', map)
# mapShow = cv2.resize(map, (1200, 1200))
# cv2.imshow('map', mapShow)
# cv2.waitKey(0)
    

