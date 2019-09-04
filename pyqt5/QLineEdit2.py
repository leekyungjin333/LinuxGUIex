import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class MyApp(QWidget):
	def __init__(self):
		super().__init__()
		
		self.initUI()
	
	def initUI(self):
		# echo_group
		self.echo_group = QGroupBox('Echo')
		self.echo_label = QLabel('Mode:')

		self.echo_cb = QComboBox()
		self.echo_cb.addItem('Normal')
		self.echo_cb.addItem('No Echo')
		self.echo_cb.addItem('Password')
		self.echo_cb.addItem('PasswordEchoOnEdit')

		self.echo_le = QLineEdit()
		self.echo_le.setPlaceholderText('Placeholder Text')
		self.echo_le.setFocus()

		# validator_group
		self.validator_group = QGroupBox('Validator')
		self.validator_label = QLabel('Type:')

		self.validator_cb = QComboBox()
		self.validator_cb.addItem('No validator')
		self.validator_cb.addItem('Integer validator')
		self.validator_cb.addItem('Double validator')

		self.validator_le = QLineEdit()
		self.validator_le.setPlaceholderText('Placeholder Text')

		# alignment_group
		self.alignment_group = QGroupBox('Alignment')
		self.alignment_label = QLabel('Type:')

		self.alignment_cb = QComboBox()
		self.alignment_cb.addItem('Left')
		self.alignment_cb.addItem('Centered')
		self.alignment_cb.addItem('Right')

		self.alignment_le = QLineEdit()
		self.alignment_le.setPlaceholderText('Placeholder Text')

		# input_mask_group
		self.input_mask_group = QGroupBox('Input mask')
		self.input_mask_label = QLabel('Type:')

		self.input_mask_cb = QComboBox()
		self.input_mask_cb.addItem('No mask')
		self.input_mask_cb.addItem('Phone number')
		self.input_mask_cb.addItem('ISO date')
		self.input_mask_cb.addItem('License key')

		self.input_mask_le = QLineEdit()
		self.input_mask_le.setPlaceholderText('Placeholder Text')

		# access_group
		self.access_group = QGroupBox('Access')
		self.access_label = QLabel('Type:')

		self.access_cb = QComboBox()
		self.access_cb.addItem('False')
		self.access_cb.addItem('True')

		self.access_le = QLineEdit()
		self.access_le.setPlaceholderText('Placeholder Text')

		# activated.connect
		self.echo_cb.activated.connect(self.echo_changed)
		self.validator_cb.activated.connect(self.validator_changed)
		self.alignment_cb.activated.connect(self.alignment_changed)
		self.input_mask_cb.activated.connect(self.input_mask_changed)
		self.access_cb.activated.connect(self.access_changed)

		# echo_layout
		self.echo_layout = QGridLayout()
		self.echo_layout.addWidget(self.echo_label, 0, 0)
		self.echo_layout.addWidget(self.echo_cb, 0, 1)
		self.echo_layout.addWidget(self.echo_le, 1, 0, 1, 2)
		self.echo_group.setLayout(self.echo_layout)

		# validator_layout
		self.validator_layout = QGridLayout()
		self.validator_layout.addWidget(self.validator_label, 0, 0)
		self.validator_layout.addWidget(self.validator_cb, 0, 1)
		self.validator_layout.addWidget(self.validator_le, 1, 0, 1, 2)
		self.validator_group.setLayout(self.validator_layout)

		# alignment_layout
		self.alignment_layout = QGridLayout()
		self.alignment_layout.addWidget(self.alignment_label, 0, 0)
		self.alignment_layout.addWidget(self.alignment_cb, 0, 1)
		self.alignment_layout.addWidget(self.alignment_le, 1, 0, 1, 2)
		self.alignment_group.setLayout(self.alignment_layout)

		# input_mask_layout
		self.input_mask_layout = QGridLayout()
		self.input_mask_layout.addWidget(self.input_mask_label, 0, 0)
		self.input_mask_layout.addWidget(self.input_mask_cb, 0, 1)
		self.input_mask_layout.addWidget(self.input_mask_le, 1, 0, 1, 2)
		self.input_mask_group.setLayout(self.input_mask_layout)

		# access_layout
		self.access_layout = QGridLayout()
		self.access_layout.addWidget(self.access_label, 0, 0)
		self.access_layout.addWidget(self.access_cb, 0, 1)
		self.access_layout.addWidget(self.access_le, 1, 0, 1, 2)
		self.access_group.setLayout(self.access_layout)

		# layout
		self.layout = QGridLayout()
		self.layout.addWidget(self.echo_group, 0, 0)
		self.layout.addWidget(self.validator_group, 1, 0)
		self.layout.addWidget(self.alignment_group, 2, 0)
		self.layout.addWidget(self.input_mask_group,0, 1)
		self.layout.addWidget(self.access_group, 1, 1)

		self.setLayout(self.layout)
	
		self.setWindowTitle('Line Editor')
		self.show()
	
	def echo_changed(self, index):
		
		if index == 0:
			self.echo_le.setEchoMode(QLineEdit.Normal)
		elif index == 1:
			self.echo_le.setEchoMode(QLineEdit.NoEcho)
		elif index == 2:
			self.echo_le.setEchoMode(QLineEdit.Password)
		elif index == 3:
			self.echo_le.setEchoMode(QLineEdit.PasswordEchoOnEdit)
	
	def validator_changed(self, index):
			
		if index == 0:
			self.validator_le.setValidator(None)
		elif index == 1:
			self.validator_le.setValidator(QIntValidator(-99, 99))
		elif index == 2:
			double_validator = QDoubleValidator(-999.0, 999.0, 2)
			double_validator.setNotation(QDoubleValidator.StandardNotation)
	
			self.validator_le.setValidator(double_validator)

		self.validator_le.clear()
	
	def alignment_changed(self, index):
		
		if index == 0:
			self.alignment_le.setAlignment(Qt.AlignLeft)
		elif index == 1:
			self.alignment_le.setAlignment(Qt.AlignCenter)
		elif index == 2:
			self.alignment_le.setAlignment(Qt.AlignRight)
	
	def input_mask_changed(self, index):
		
		if index == 0:
			self.input_mask_le.setInputMask('')
		elif index == 1:
			self.input_mask_le.setInputMask('000-0000-0000')
		elif index == 2:
			self.input_mask_le.setInputMask('0000-00-00')
			self.input_mask_le.setText('20190410')
			self.input_mask_le.setCursorPosition(0)
		elif index == 3:
			self.input_mask_le.setInputMask('>AAAAA-AAAAA-AAAAA-AAAAA;#')
			
	def access_changed(self, index):
		
		if index == 0:
			self.access_le.setReadOnly(False)
		elif index == 1:
			self.access_le.setReadOnly(True)


if __name__ == '__main__':
	
	app = QApplication(sys.argv)
	ex = MyApp()
	sys.exit(app.exec_())
