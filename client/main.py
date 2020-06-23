from MainWindow import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QSystemTrayIcon, QAction
from PyQt5.QtWidgets import QMessageBox, QMenu, QDesktopWidget, QLabel, QPushButton
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QEvent, pyqtSignal, pyqtSlot, QRect, Qt,QThread
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen,QCursor
import sys
from system_hotkey import SystemHotkey
import cv2
import sys
sys.path.append("..")
from capture import capture
from translator import trans_img

class SystemTray(object):
	# 程序托盘类
	def __init__(self, w):
		self.app = app
		self.w = w
		QApplication.setQuitOnLastWindowClosed(False)  # 禁止默认的closed方法，只能使用qapp.quit()的方法退出程序
		self.tray = QSystemTrayIcon(self.w)
		self.initUI()
		self.run()

	def initUI(self):
		# 设置托盘图标
		self.tray.setIcon(QIcon('icon/tray.ico'))

	def quitApp(self):
		# 退出程序
		re = QMessageBox.question(self.w, "提示", "退出系统", QMessageBox.Yes |
								  QMessageBox.No, QMessageBox.No)
		if re == QMessageBox.Yes:
			self.tray.setVisible(False)  # 隐藏托盘控件，托盘图标刷新不及时，提前隐藏
			app.quit()  # 退出程序

	def capture(self):
		print('capture')
		pass

	def run(self):
		a1 = QAction('&截图   Ctrl 1', triggered=self.capture)
		a2 = QAction('&退出', triggered=self.quitApp)

		_translate = QtCore.QCoreApplication.translate

		trayMenu = QMenu()
		trayMenu.addAction(a1)
		trayMenu.addAction(a2)
		self.tray.setContextMenu(trayMenu)
		self.tray.show()  # 不调用show不会显示系统托盘消息，图标隐藏无法调用

		# 信息提示
		sys.exit(self.app.exec_())  # 持续对app的连接


class MyLabel(QLabel):
	x0 = 0
	y0 = 0
	x1 = 0
	y1 = 0
	loc = []
	flag = False
	msg = pyqtSignal(int,int,int,int)

	def __init__(self, parent=None):
		super(MyLabel, self).__init__(parent)
	# 鼠标点击事件
	def mousePressEvent(self, event):
		self.flag = True
		self.x0 = event.x()
		self.y0 = event.y()

	# 鼠标释放事件
	def mouseReleaseEvent(self, event):
		self.flag = False
		if len(self.loc) == 4:
			self.sendmsg(self.loc[0],self.loc[1],self.loc[2],self.loc[3])
			self.loc = []

	# 鼠标移动事件
	def mouseMoveEvent(self, event):
		if self.flag:
			self.x1 = event.x()
			self.y1 = event.y()
			self.update()

	# 绘制事件
	def paintEvent(self, event):
		super().paintEvent(event)
		if self.flag:
			if self.x1 < self.x0:
				self.loc = [self.x1, self.y1,abs(self.x1 - self.x0), abs(self.y1 - self.y0)]
				rect = QRect(self.loc[0],self.loc[1],self.loc[2],self.loc[3])
				painter = QPainter(self)
				painter.setPen(QPen(Qt.blue, 1, Qt.SolidLine))
				painter.drawRect(rect)
				print( self.x1, self.y1,abs(self.x1 - self.x0), abs(self.y1 - self.y0))
				# self.sendmsg( self.x1, self.y1,abs(self.x1 - self.x0), abs(self.y1 - self.y0))
			else:
				self.loc = [self.x0, self.y0, abs(self.x1 - self.x0),  abs(self.y1 - self.y0)]
				print(self.loc)
				rect = QRect(self.loc[0],self.loc[1],self.loc[2],self.loc[3])
				painter = QPainter(self)
				painter.setPen(QPen(Qt.blue, 1, Qt.SolidLine))
				painter.drawRect(rect)
				print(self.loc)


	def sendmsg(self,x0,y0,w,h):
		# print(x0)
		self.msg.emit(x0,y0,w,h)


class MyMainWindow(QMainWindow, Ui_MainWindow, QObject):
	# 定义一个热键信号
	sig_keyhot = pyqtSignal(str)

	def __init__(self, parent=None):
		super(MyMainWindow, self).__init__(parent)
		#去掉边框|任务栏|窗口置顶
		self.setWindowFlags(Qt.FramelessWindowHint|Qt.Tool|Qt.WindowStaysOnTopHint)
		self.setupUi(self)

		self.screenWidget = QtWidgets.QWidget()
		self.screenWidget.setObjectName("screenWidget")
		self.screenWidget.setWindowFlags(Qt.FramelessWindowHint)

		self.centralWidget = self.findChild(QtWidgets.QWidget,'centralwidget')
		self.centralWidget.setWindowFlags(Qt.FramelessWindowHint)

		self.toolWidget = self.findChild(QtWidgets.QWidget,'toolWidget')
		self.toolWidget.setWindowFlags(Qt.FramelessWindowHint)

		#设置按钮的槽函数
		self.btn_i2u.clicked.connect(lambda :self.i2u())
		btn_bdfy = self.findChild(QtWidgets.QToolButton, 'baidu')
		btn_i2t = self.findChild(QtWidgets.QToolButton, 'img2txt')
		btn_copy = self.findChild(QtWidgets.QToolButton, 'copy')

		#背景透明
		self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

		# 2. 设置我们的自定义热键响应函数
		self.sig_keyhot.connect(self.MKey_pressEvent)

		# 3. 初始化两个热键
		self.hk_start, self.hk_stop = SystemHotkey(), SystemHotkey()
		# 4. 绑定快捷键和对应的信号发送函数
		self.hk_start.register(('control', '1'), callback=lambda x: self.send_key_event("0"))
		self.hk_stop.register(('control', '2'), callback=lambda x: self.send_key_event("1"))

		esc = QAction('', triggered=self.esc)
		esc.setShortcut('esc')
		self.addAction(esc)
		self.screenWidget.addAction(esc)
		# 系统托盘
		self.systemtray = SystemTray(self)


	def i2u(self):

		print('start i2u')
		res_im,res_txt = trans_img.generate_trans_img(img=self.cap_im.copy(),method = 'local')
		print(res_txt)
		cv2.imshow('',res_im)
		cv2.waitKey(0)

	@pyqtSlot()
	def esc(self):
		self.screenWidget.hide()
		self.hide()
		import sip
		try:
			self.screenLabel.deleteLater()
			sip.delete(self.screenLabel)
		except Exception as e:
			print('error')
			pass
		try:
			self.imgLabel.deleteLater()
			sip.delete(self.imgLabel)
		except Exception as e:
			print('delete imgLabel error')


	#装饰器接受的数据类型必须和信号量的数据类型保持一致
	@pyqtSlot(int,int,int,int)
	def capture(self, x0, y0, w, h):
		y1= y0+h
		x1= x0+w
		print(x0, y0, x1,y1)
		#必须用copy，否则在使用im.data的时候会报错
		self.cap_im = self.im[y0:y1,x0:x1].copy()
		height, width, depth = self.cap_im.shape

		self.esc()
		self.imgLabel = QLabel(self.centralWidget)
		self.imgLabel.resize(width, height)

		qimg = QImage(self.cap_im.data, width, height, width * depth, QImage.Format_RGB888)
		pixmap = QtGui.QPixmap(qimg).scaled(self.imgLabel.width(), self.imgLabel.height())
		self.imgLabel.setPixmap(pixmap)
		self.imgLabel.move(0,0)

		self.toolWidget.move(0,height-30)
		self.centralWidget.raise_()
		self.toolWidget.raise_()

		self.resize(width,height)
		self.move(x0,y0)
		self.show()

	# 热键处理函数
	def MKey_pressEvent(self, i_str):
		if i_str == '0':
			im = capture.capture()
			height, width, depth = im.shape
			self.im = im
			self.screenLabel = MyLabel(self.screenWidget)
			self.screenLabel.resize(width, height)
			self.screenLabel.msg.connect(self.capture)
			print(self.screenLabel.width(), self.screenLabel.height())
			qimg = QImage(im.data, width, height, width * depth, QImage.Format_RGB888)
			pixmap = QtGui.QPixmap(qimg).scaled(self.screenLabel.width(), self.screenLabel.height())
			self.screenLabel.setPixmap(pixmap)
			self.screenWidget.move(0,0)
			self.screenWidget.showFullScreen()

	# 热键信号发送函数(将外部信号，转化成qt信号)
	def send_key_event(self, i_str):
		self.sig_keyhot.emit(i_str)
		print(i_str)


	#设置无边框移动
	def mousePressEvent(self, event):
		if event.button() == Qt.LeftButton:
			self.m_flag = True
			self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
			event.accept()
			self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标


	def mouseMoveEvent(self, QMouseEvent):
		if Qt.LeftButton and self.m_flag:
			self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
			QMouseEvent.accept()


	def mouseReleaseEvent(self, QMouseEvent):
		self.m_flag = False
		self.setCursor(QCursor(Qt.ArrowCursor))


if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWin = MyMainWindow()
	sys.exit(app.exec_())
