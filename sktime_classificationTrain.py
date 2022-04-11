import random
import sys
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from classification_train import Ui_MainWindow
from PySide2 import QtWidgets, QtGui
import glob
import pyqtgraph as pg

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.pushButton.pressed.connect(self.GetTrainFolder)
        self.pushButton_3.pressed.connect(self.GetTestFolder)
        self.pushButton_2.pressed.connect(self.ReadTrainTestFolder)
        self.listWidget.itemDoubleClicked.connect(self.onDoubleClick)

    def GetTrainFolder(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.lineEdit.setText('{}'.format(directory))

    def GetTestFolder(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.lineEdit_2.setText('{}'.format(directory))

    def ReadTrainTestFolder(self):
        train_files = glob.glob(self.lineEdit.text() + "\\*.csv")
        test_files = glob.glob(self.lineEdit_2.text() + "\\*.csv")  
        self.listWidget.addItems(train_files)
        self.listWidget_2.addItems(test_files)

    def onDoubleClick(self, item):
        print(item.text())    
             
        hour = [1,2,3,4,5,6,7,8,9,10]
        temperature = [30,32,34,32,33,31,29,32,35,45]        
        pg.plot(hour, temperature)
      
        

app = QApplication(sys.argv)
w = MainWindow()
app.exec_()
