import os

imageFolderPath = 'NG(BMP)'

fileList = [_.rstrip(".BMP") for _ in os.listdir(imageFolderPath) if _.endswith('.BMP')]

# coorList = []
# for i in range(0, 4992, 128):
#     for j in range(0, 4992, 128):
#         coorList.append((i, j))

# for fileName in fileList:
#     with open(f'{os.path.join(imageFolderPath, fileName)}.txt', 'w') as txtFile:
#         for coor in coorList:
#             txtFile.write(f'0 ')
#             txtFile.write(f'{(coor[0]+128)/5120} ')
#             txtFile.write(f'{(coor[1]+128)/5120} ')
#             txtFile.write(f'{256/5120} ')
#             txtFile.write(f'{256/5120}\n')

for fileName in fileList:
    with open(f'{os.path.join(imageFolderPath, fileName)}.txt', 'r') as txtFile:
        txtList = txtFile.readlines()
    with open(f'{os.path.join(imageFolderPath, fileName)}.txt', 'w') as txtFile:
        for txtLine in txtList:
            txtIndex = txtLine.replace('\n', '').split(' ')
            txtFile.write(f'{int(txtIndex[0])} ')
            txtFile.write(f'{int(float(txtIndex[1])*5120-128)} ')
            txtFile.write(f'{int(float(txtIndex[2])*5120-128)} ')
            txtFile.write(f'{int(float(txtIndex[1])*5120+128)} ')
            txtFile.write(f'{int(float(txtIndex[2])*5120+128)}\n')