import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLineEdit

class MyApp(QWidget):

	def __init__(self):
		super().__init__()
		self.LineEdit = QLineEdit()
		self.initUI()

	def clicked_button(self):
		argc=self.sender().text()
		a = self.LineEdit.text()
		if argc == '=':
			if a[-1] == '+' or a[-1] == '-' or a[-1] == '*' or a[-1] == '/':
				a = a[:-1]
			self.LineEdit.setText(str(eval(a)))
		else:
#			if a[-1] == '+' or a[-1] == '-' or a[-1] == '*' or a[-1] == '/':
#				self.LineEdit.setText(a)
			self.LineEdit.setText(a+argc)

	def clicked_buttonDEL(self):
		a = self.LineEdit.text()
		if a[-3:] == ' + ' :
			self.LineEdit.setText(a[:-3])
		elif a != '0' :
			self.LineEdit.setText(a[:-1])
	
	def clicked_buttonC(self):
		self.LineEdit.setText('')

	def initUI(self):
		self.LineEdit.setPlaceholderText("0")
		self.LineEdit.setReadOnly(True)
		self.LineEdit.setAlignment(Qt.AlignRight)

		button7 = QPushButton('7')
		button7.clicked.connect(self.clicked_button)
		button8 = QPushButton('8')
		button8.clicked.connect(self.clicked_button)
		button9 = QPushButton('9')
		button9.clicked.connect(self.clicked_button)

		button4 = QPushButton('4')
		button4.clicked.connect(self.clicked_button)
		button5 = QPushButton('5')
		button5.clicked.connect(self.clicked_button)
		button6 = QPushButton('6')
		button6.clicked.connect(self.clicked_button)

		button1 = QPushButton('1')
		button1.clicked.connect(self.clicked_button)
		button2 = QPushButton('2')
		button2.clicked.connect(self.clicked_button)
		button3 = QPushButton('3')
		button3.clicked.connect(self.clicked_button)

		button0 = QPushButton('0')
		button0.clicked.connect(self.clicked_button)
		buttonDEL = QPushButton('DEL')
		buttonDEL.clicked.connect(self.clicked_buttonDEL)
		buttonC = QPushButton('C')
		buttonC.clicked.connect(self.clicked_buttonC)

		buttonAdd = QPushButton('+')
		buttonAdd.clicked.connect(self.clicked_button)
		buttonSub = QPushButton('-')
		buttonSub.clicked.connect(self.clicked_button)
		buttonMul = QPushButton('*')
		buttonMul.clicked.connect(self.clicked_button)
		buttonDiv = QPushButton('/')
		buttonDiv.clicked.connect(self.clicked_button)
		buttonE = QPushButton('=')
		buttonE.clicked.connect(self.clicked_button)

		hbox1 = QHBoxLayout()
		hbox1.addStretch(1)
		hbox1.addWidget(button7)
		hbox1.addWidget(button8)
		hbox1.addWidget(button9)
		hbox1.addStretch(1)

		hbox2 = QHBoxLayout()
		hbox2.addStretch(1)
		hbox2.addWidget(button4)
		hbox2.addWidget(button5)
		hbox2.addWidget(button6)
		hbox2.addStretch(1)

		hbox3 = QHBoxLayout()
		hbox3.addStretch(1)
		hbox3.addWidget(button1)
		hbox3.addWidget(button2)
		hbox3.addWidget(button3)
		hbox3.addStretch(1)

		hbox4 = QHBoxLayout()
		hbox4.addStretch(1)
		hbox4.addWidget(button0)
		hbox4.addWidget(buttonDEL)
		hbox4.addWidget(buttonC)
		hbox4.addStretch(1)
		
		vbox1 = QVBoxLayout()
		vbox1.addLayout(hbox1)
		vbox1.addLayout(hbox2)
		vbox1.addLayout(hbox3)
		vbox1.addLayout(hbox4)

		vbox2 = QVBoxLayout()
		vbox2.addWidget(buttonAdd)
		vbox2.addWidget(buttonSub)
		vbox2.addWidget(buttonMul)
		vbox2.addWidget(buttonDiv)
		vbox2.addWidget(buttonE)

		hbox5 = QHBoxLayout()
		hbox5.addLayout(vbox1)
		hbox5.addLayout(vbox2)

		vbox3 = QVBoxLayout()
		vbox3.addStretch(1)
		vbox3.addWidget(self.LineEdit)
		vbox3.addLayout(hbox5)
		vbox3.addStretch(1)


		self.setLayout(vbox3)
		
		self.setWindowTitle('calculator')
		self.setGeometry(300, 300, 300, 300)
		self.show()		


if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = MyApp()
	sys.exit(app.exec_())
