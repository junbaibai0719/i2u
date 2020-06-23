# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(574, 447)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.toolWidget = QtWidgets.QWidget(self.centralwidget)
        self.toolWidget.setGeometry(QtCore.QRect(9, 423, 226, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolWidget.sizePolicy().hasHeightForWidth())
        self.toolWidget.setSizePolicy(sizePolicy)
        self.toolWidget.setObjectName("toolWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.toolWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_local = QtWidgets.QToolButton(self.toolWidget)
        self.btn_local.setCheckable(False)
        self.btn_local.setObjectName("btn_local")
        self.horizontalLayout.addWidget(self.btn_local)
        self.btn_baidu = QtWidgets.QToolButton(self.toolWidget)
        self.btn_baidu.setText("")
        self.btn_baidu.setObjectName("btn_baidu")
        self.horizontalLayout.addWidget(self.btn_baidu)
        self.btn_img2txt = QtWidgets.QToolButton(self.toolWidget)
        self.btn_img2txt.setObjectName("btn_img2txt")
        self.horizontalLayout.addWidget(self.btn_img2txt)
        self.btn_i2u = QtWidgets.QToolButton(self.toolWidget)
        self.btn_i2u.setObjectName("btn_i2u")
        self.horizontalLayout.addWidget(self.btn_i2u)
        self.btn_i2u.raise_()
        self.btn_local.raise_()
        self.btn_baidu.raise_()
        self.btn_img2txt.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_local.setToolTip(_translate("MainWindow", "本地翻译"))
        self.btn_local.setText(_translate("MainWindow", "本地翻译"))
        self.btn_baidu.setToolTip(_translate("MainWindow", "百度翻译"))
        self.btn_img2txt.setToolTip(_translate("MainWindow", "提取文字"))
        self.btn_img2txt.setText(_translate("MainWindow", "提取文字"))
        self.btn_i2u.setToolTip(_translate("MainWindow", "i2u翻译"))
        self.btn_i2u.setText(_translate("MainWindow", "远程翻译"))
