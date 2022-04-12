import random
import sys
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
from classification_train import Ui_MainWindow
from PySide2 import QtWidgets, QtGui
import glob
import pyqtgraph as pg
import pandas as pd
import os
import numpy as np
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.reduce import Tabularizer
from sklearn.ensemble import RandomForestClassifier
from sktime.classification.kernel_based import RocketClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sktime.classification.interval_based import (
    CanonicalIntervalForest,
    DrCIF,
    RandomIntervalSpectralEnsemble,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.pushButton.pressed.connect(self.GetTrainFolder)
        self.pushButton_3.pressed.connect(self.GetTestFolder)
        self.pushButton_2.pressed.connect(self.ReadTrainTestFolder)
        self.listWidget.itemDoubleClicked.connect(self.onDoubleClick)
        self.listWidget_2.itemDoubleClicked.connect(self.onDoubleClick)
        self.horizontalSlider.valueChanged.connect(self.sliderChangedValue)
        self.comboBox.addItems([self.tr('RandomForestClassifier'),
                                self.tr('RocketClassifier'),
                                self.tr('TimeSeriesForestClassifier'),
                                self.tr('RandomIntervalSpectralEnsemble'),
                                self.tr('SupervisedTimeSeriesForest'),
                                ])
        self.pushButton_5.pressed.connect(self.SetModelOutputFolder)
        self.pushButton_4.pressed.connect(self.StartTrain)

    def SetModelOutputFolder(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.lineEdit_3.setText('{}'.format(directory))

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
        df = pd.read_csv(item.text(), index_col=None, header=None,usecols=[2])        
        pg.plot(df.index.values, df.iloc[:, 0], title = os.path.basename(item.text()))
    
    def sliderChangedValue(self):          
        train_value = self.horizontalSlider.value()
        test_value = 100 - train_value
        self.label_5.setText(str(train_value))
        self.label_6.setText(str(test_value))

    def StartTrain(self): 
        train_files = glob.glob(self.lineEdit.text() + "\\*.csv")
        test_files = glob.glob(self.lineEdit_2.text() + "\\*.csv")
        train_li = []
        for filename in train_files:                
            df = pd.read_csv(filename, index_col=None, header=None,usecols=[2])
            train_li.append(df)
        X_df = pd.concat(train_li, axis=1, ignore_index=True)
        X_df = X_df.T  

        test_li = []        
        for filename in test_files:
            df = pd.read_csv(filename, index_col=None, header=None,usecols=[2])
            test_li.append(df)
        X_df_ng = pd.concat(test_li, axis=1, ignore_index=True)
        X_df_ng = X_df_ng.T

        X_df = X_df.append(X_df_ng)           
        X_df_tab = from_2d_array_to_nested(X_df)

        Y_df_ok = np.zeros(len(test_li), dtype="int32")
        Y_df_ng = np.ones(len(train_li), dtype="int32")
        Y_df = np.concatenate([Y_df_ok, Y_df_ng], 0)
        
        X_train, X_test, y_train, y_test = train_test_split(X_df_tab, Y_df, test_size= (100 - self.horizontalSlider.value()) / 100)
        self.tableWidget.setRowCount(0)
        selectedModel = self.comboBox.currentText()
        if(selectedModel == "RandomForestClassifier"):
            classifier = make_pipeline(Tabularizer(), RandomForestClassifier())
            classifier.fit(X_train, y_train)
            self.lineEdit_5.setText(str(classifier.score(X_train, y_train)))  
            self.lineEdit_6.setText(str(classifier.score(X_test, y_test)))          
            for i in range(len(X_test)): 
                row = self.tableWidget.rowCount()
                self.tableWidget.setRowCount(row)                
                classifier_preds = classifier.predict(X_test.iloc[i].to_frame())                
                self.addTableRow(self.tableWidget, [str(i),str(y_test[i]), str(classifier_preds)])   
            
        elif(selectedModel == "RocketClassifier"):
            rocket = RocketClassifier()
            rocket.fit(X_train, y_train)
            self.lineEdit_5.setText(str(rocket.score(X_train, y_train)))  
            self.lineEdit_6.setText(str(rocket.score(X_test, y_test))) 
            for i in range(len(X_test)): 
                row = self.tableWidget.rowCount()
                self.tableWidget.setRowCount(row)                
                rocket_preds = rocket.predict(X_test.iloc[i].to_frame())                
                self.addTableRow(self.tableWidget, [str(i),str(y_test[i]), str(rocket_preds)]) 
        
        elif(selectedModel == "TimeSeriesForestClassifier"):
            tsf = TimeSeriesForestClassifier(n_estimators=50, random_state=47)
            tsf.fit(X_train, y_train)
            self.lineEdit_5.setText(str(tsf.score(X_train, y_train)))  
            self.lineEdit_6.setText(str(tsf.score(X_test, y_test))) 
            for i in range(len(X_test)): 
                row = self.tableWidget.rowCount()
                self.tableWidget.setRowCount(row)                
                tsf_preds = tsf.predict(X_test.iloc[i].to_frame())                
                self.addTableRow(self.tableWidget, [str(i),str(y_test[i]), str(tsf_preds)]) 

        elif(selectedModel == "RandomIntervalSpectralEnsemble"):
            rise = RandomIntervalSpectralEnsemble(n_estimators=50, random_state=47)
            rise.fit(X_train, y_train)
            self.lineEdit_5.setText(str(rise.score(X_train, y_train)))  
            self.lineEdit_6.setText(str(rise.score(X_test, y_test))) 
            for i in range(len(X_test)): 
                row = self.tableWidget.rowCount()
                self.tableWidget.setRowCount(row)                
                rise_preds = rise.predict(X_test.iloc[i].to_frame())                
                self.addTableRow(self.tableWidget, [str(i),str(y_test[i]), str(rise_preds)]) 

        elif(selectedModel == "SupervisedTimeSeriesForest"):
            stsf = SupervisedTimeSeriesForest(n_estimators=50, random_state=47)
            stsf.fit(X_train, y_train)
            self.lineEdit_5.setText(str(stsf.score(X_train, y_train)))  
            self.lineEdit_6.setText(str(stsf.score(X_test, y_test))) 
            for i in range(len(X_test)): 
                row = self.tableWidget.rowCount()
                self.tableWidget.setRowCount(row)                
                stsf_preds = rise.predict(X_test.iloc[i].to_frame())                
                self.addTableRow(self.tableWidget, [str(i),str(y_test[i]), str(stsf_preds)]) 
        else:
            print("None")

    def addTableRow(self, table, row_data):
        row = table.rowCount()
        table.setRowCount(row+1)
        col = 0
        for item in row_data:
            cell = QTableWidgetItem(str(item))
            table.setItem(row, col, cell)
            col += 1

app = QApplication(sys.argv)
w = MainWindow()
app.exec_()
