import  win32gui, win32ui, win32con, win32api
from datetime import datetime
import numpy as np
import cv2

path = ''

def log(*arg, fn: str = ''):
	if fn == '':
		fn = './log.txt'
	sentence = ''.join(arg)
	nowTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	with open(fn, 'a') as f:
		f.writelines(sentence + '\t%s\n' % nowTime)
	print(sentence)

def capture(x0=0, y0=0, w=0, h=0,hwnd = 0):
	try:
		# 返回句柄窗口的设备环境、覆盖整个窗口，包括非客户区，标题栏，菜单，边框
		hwndDC = win32gui.GetWindowDC(hwnd)

		# 创建设备描述表
		mfcDC = win32ui.CreateDCFromHandle(hwndDC)

		# 创建内存设备描述表
		saveDC = mfcDC.CreateCompatibleDC()

		# 创建位图对象
		saveBitMap = win32ui.CreateBitmap()

		if w==0 and h == 0:
			MoniterDev = win32api.EnumDisplayMonitors(None, None)
			w = MoniterDev[0][2][2]
			h = MoniterDev[0][2][3]

		saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
		saveDC.SelectObject(saveBitMap)
		# 截图至内存设备描述表
		img_dc = mfcDC
		mem_dc = saveDC
		mem_dc.BitBlt((0, 0), (w, h), img_dc, (x0, y0), win32con.SRCCOPY)
		signedIntsArray = saveBitMap.GetBitmapBits(True)
		img = np.frombuffer(signedIntsArray, dtype=np.uint8)
		img = np.reshape(img, (h, w, 4))
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) #rgba to rgb
		# cv2.imshow("OpenCV", img)
		# cv2.waitKey()
		# 将截图保存到文件中
		#             saveBitMap.SaveBitmapFile(mem_dc, path \
		#                                       + '/%s.bmp' % hwnd)

		# 释放内存，不然会造成资源泄漏
		win32gui.DeleteObject(saveBitMap.GetHandle())
		saveDC.DeleteDC()
		log("截图成功,path:%s" % (path), fn='./capture_log.txt')
		return img
	except Exception as e:
		log("截图失败,%s,%s" % (e, path), fn='./capture_log.txt')

