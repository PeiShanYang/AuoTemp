import os, csv
import time      



class UsingCsv():
    """
    Operations with csv
    """
    def __init__(self, fileName: str, path: str, title: list=None):
        """create csv

        Args:
            fileName (str): filename
            path (str): written path
            title (list):  title of csv
        """
        super(UsingCsv, self).__init__()
        self.title = title
        self.path = path
        self.file = os.path.join(path, fileName)
        self.create_folder()
        
    def create_folder(self):
        """create written path
        """
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def create_csv(self):
        """create csv file
        """
        with open(self.file, 'w', newline='') as csvfile:
            if self.title != None:
                writer = csv.writer(csvfile)
                writer.writerow(self.title)  
            csvfile.close()
    
    def writing(self, result:list, mode='w'): 
        """write out csv

        Args:
            result (list): result to write
            mode (str, optional): Defaults to 'w'.
                'w': is single write
                'a': is continue to write
        """
        with open(self.file, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(result)
            csvfile.close()
