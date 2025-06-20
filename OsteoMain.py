# -*- coding: utf-8 -*-
"""
@author: Stanislav Pavlovich Gamenyuk
"""

import sys
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import pathlib
from pathlib import Path
import ast
from glob import glob
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRect, QEvent, QUrl, QPoint, Qt
from PyQt5.QtGui import QIcon, QPixmap, QPicture, QColor, QImage
from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QAction, QFileDialog, QApplication, QMessageBox, QPushButton, QLabel, QFrame, QMenu, QStatusBar, QGridLayout)
import OsteoDesign

class ExampleApp(QtWidgets.QMainWindow, OsteoDesign.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.openAction.triggered.connect(self.openFile)
        #self.saveAction.triggered.connect("")
        self.exitAction.triggered.connect(self.onExit)
        self.loadBttn.clicked.connect(self.loadImage)
        
        #Название
        title = 'Osteo Digital Classifier v25 ©Created by Stanislav Pavlovich Gamenyuk'
        self.setWindowTitle(title)
        
        #Заставка
        pixmap = QPixmap('')
        self.imgLbl.setPixmap(pixmap)
        
        self.work_dir = os.getcwd() # Получение рабочей директории через метод getcwd
        self.WidgetMes = QLabel("") # Создание виджета-сообщения для дальнейшего использования в статус-баре
        self.nf = np.float32
        self.objList = []

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, "Открыть файл", self.work_dir,
                                            "Графика (*.jpg *.jpeg *.bmp *.png *.tiff)\n Все файлы (*.*)")[0]
        if (fname == ""):
            return
        self.file_name = fname
        self.openImageFile()
        
        # Статусбар с именем файла
        url = QUrl.fromLocalFile(fname)
        self.WidgetMes.setText("Файл: {}".format(url.fileName()))
        self.statusbar.addPermanentWidget(self.WidgetMes)
        
    def openImageFile(self):
        if (self.file_name == ""):
            return
        self.imgLbl.setPixmap(QPixmap(self.file_name))
        self.objName.setText("")
        
    def loadImage(self):
        path = self.file_name
        files = sorted(glob(str(path)))
        for file_name in files:
            img_load = cv.imread(file_name)
            height = 224
            width = 224
            dim = (width, height)
            img_load = cv.resize(img_load, dim, interpolation = cv.INTER_AREA)
            img_load.astype('float32')
            self.objList.append(img_load)
            self.proModel(self.objList)
        
    def proModel(self, objList):
        dir_path = Path(pathlib.Path.cwd(), 'OsteoNet.keras')
        #print(dir_path)
        model = tf.keras.models.load_model(dir_path)
        sample = tf.expand_dims(objList[-1], 0)
        sample = tf.convert_to_tensor(sample)
        pred = model.predict(sample)
        
        labels = ['Норма', 'Остеопения', 'Остеопороз']
        class_names = labels
        
        score = tf.nn.softmax(pred[0])
        self.class_names = str(class_names[np.argmax(score)])
        self.objName.setText(self.class_names)

    def onExit(self):
        self.close()
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()
    
if __name__ == '__main__':
    main()