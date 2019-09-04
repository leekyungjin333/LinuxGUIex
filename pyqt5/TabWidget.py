import sys
from PyQt5.QtWidgets import *


class MyApp(QDialog):

	def __init__(self):
		super().__init__()

		self.initUI()

	def initUI(self):

		tabs = QTabWidget()
		tabs.addTab(FirstTab(), 'First')
		tabs.addTab(SecondTab(), 'Second')

		buttonbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
		buttonbox.accepted.connect(self.accept)
		buttonbox.rejected.connect(self.reject)

		vbox = QVBoxLayout()
		vbox.addWidget(tabs)
		vbox.addWidget(buttonbox)

		self.setLayout(vbox)

		self.setWindowTitle('QTabWidget')
		self.setGeometry(300, 300, 400, 300)
		self.show()


class FirstTab(QWidget):

	def __init__(self):
		super().__init__()

		self.initUI()

	def initUI(self):

		name = QLabel('length:')
		self.LE = QLineEdit()
		cb = QComboBox(self)
		cb.addItem('mm')
		cb.addItem('cm')
		cb.addItem('m')
		cb.addItem('km')
		self.Ctext=cb.currentText()
		self.Cnum=self.lengthatoi(self.Ctext)
		cb.activated[str].connect(self.onActivated)

		hbox = QHBoxLayout()
		hbox.addWidget(name)
		hbox.addWidget(self.LE)
		hbox.addWidget(cb)
		hbox.addStretch()

		self.setLayout(hbox)

	def onActivated(self, text):
		aa = self.LE.text()
		num=self.lengthatoi(text)
		self.LE.setText(str(float(aa)/(num/self.Cnum)))
		self.Cnum = num
		self.LE.adjustSize()

	def lengthatoi(self, arg):
		if arg == 'mm':
			return 1
		elif arg == 'cm':
			return 10
		elif arg == 'm':
			return 1000
		elif arg == 'km':
			return 1000000;


class SecondTab(QWidget):

	def __init__(self):
		super().__init__()

		self.initUI()

	def initUI(self):

		name = QLabel('weight:')
		self.LE = QLineEdit()
		cb = QComboBox(self)
		cb.addItem('mg')
		cb.addItem('g')
		cb.addItem('kg')
		cb.addItem('t')
		self.Ctext=cb.currentText()
		self.Cnum=self.lengthatoi(self.Ctext)
		cb.activated[str].connect(self.onActivated)

		hbox = QHBoxLayout()
		hbox.addWidget(name)
		hbox.addWidget(self.LE)
		hbox.addWidget(cb)
		hbox.addStretch()

		self.setLayout(hbox)

	def onActivated(self, text):
		aa = self.LE.text()
		num=self.lengthatoi(text)
		self.LE.setText(str(float(aa)/(num/self.Cnum)))
		self.Cnum = num
		self.LE.adjustSize()

	def lengthatoi(self, arg):
		if arg == 'mg':
			return 1
		elif arg == 'g':
			return 1000
		elif arg == 'kg':
			return 1000000
		elif arg == 't':
			return 1000000000;


class ThirdTab(QWidget):

	def __init__(self):
		super().__init__()

		self.initUI()

	def initUI(self):

		lbl = QLabel('Terms and Conditions')
		text_browser = QTextBrowser()
		text_browser.setText('This is the terms and conditions')
		checkbox = QCheckBox('Check the terms and conditions.')

		vbox = QVBoxLayout()
		vbox.addWidget(lbl)
		vbox.addWidget(text_browser)
		vbox.addWidget(checkbox)

		self.setLayout(vbox)


if __name__ == '__main__':

	app = QApplication(sys.argv)
	ex = MyApp()
	sys.exit(app.exec_())
