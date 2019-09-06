import sys
import os
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt
from PIL import Image, ImageOps

class Button(QToolButton):
    def __init__(self, text, parent=None):
        super(Button, self).__init__(parent)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setText(text)

    def sizeHint(self):
        size = super(Button, self).sizeHint()
        size.setHeight(size.height() + 20)
        size.setWidth(max(size.width(), size.height()))
        return size  


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        onlyInt = QIntValidator()
        self.Rs = QCheckBox('Resize', self)
        self.Rt = QCheckBox('Rotate', self)
        self.Hf = QCheckBox('Hflip', self)
        self.Vf = QCheckBox('Vflip', self)
        self.Rn = QCheckBox('Rename', self)
        self.RsEW = QLineEdit()
        self.RsEW.setPlaceholderText('Width')
        self.RsEW.setValidator(onlyInt)
        self.RsEH = QLineEdit()
        self.RsEH.setPlaceholderText('Height')
        self.RsEH.setValidator(onlyInt)
        self.RtE = QLineEdit()
        self.RtE.setValidator(onlyInt)
        RnP = QLabel('Prefix')
        RnS = QLabel('Suffix')
        self.Run = self.createButton("RUN", self.RunClicked)
        self.RnPE = QLineEdit()
        self.RnSE = QLineEdit('0')
        self.RnSE.setValidator(onlyInt)
        Num = QLabel('Number')
        hbox=QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(Num)
        Path = QLabel('PATH')
        self.PE = QLineEdit('./trainset/')

        grid = QGridLayout()
        grid.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(grid)
        
        grid.addWidget(self.Rs, 0, 0, 1, 1)
        grid.addWidget(self.Rt, 1, 0, 1, 1)
        grid.addWidget(self.Hf, 2, 0, 1, 1)
        grid.addWidget(self.Vf, 3, 0, 1, 1)
        grid.addWidget(self.Rn, 4, 0, 1 ,1)
        grid.addWidget(self.RsEW, 0, 1, 1, 1)
        grid.addWidget(self.RsEH, 0, 2, 1, 1)
        grid.addWidget(self.RtE, 1, 1, 1, 1)
        grid.addWidget(self.Run, 0, 3, 2, 1)
        grid.addWidget(RnP, 4, 1, 1, 1)
        grid.addWidget(RnS, 4, 3, 1, 1)
        grid.addWidget(self.RnPE, 5, 1, 1, 1)
        grid.addWidget(self.RnSE, 5, 3, 1, 1)
        grid.addLayout(hbox, 5, 2, 1, 1)
        grid.addWidget(Path, 6, 0, 1, 1)
        grid.addWidget(self.PE, 6, 1, 1, 3)
        
        self.setWindowTitle('Augmentation Application')
        self.setGeometry(300, 300, 300, 200)
        self.show()

    def RunClicked(self):
        CRs = self.Rs.isChecked()
        CRt = self.Rt.isChecked()
        CHf = self.Hf.isChecked()
        CVf = self.Vf.isChecked()
        CRn = self.Rn.isChecked()
        self.FileAugment(CRs, CRt, CHf, CVf, CRn)


    def FileAugment(self, CRs, CRt, CHf, CVf, CRn):
        path_dir = self.PE.text()
        file_list = os.listdir(path_dir)
        #print(file_list)
        #print(len(file_list))

        for fname in file_list:
            if fname[-3:] == 'jpg' or fname[-4:] == 'jpeg':
                img = Image.open(path_dir+fname)
                savePath = './Augmentation_trainset/'
                if CRs == True:
                    img = img.resize((int(self.RsEW.text()), int(self.RsEH.text())))
                    fname = 'Rs_'+fname
                    savePath = savePath + '_RS'
                if CRt == True:
                    img = img.rotate(int(self.RtE.text()))
                    fname = 'RT_'+fname
                    savePath = savePath + '_RT'
                if CHf == True:
                    img = ImageOps.mirror(img)
                    fname = 'HF_'+fname
                    savePath = savePath + '_HF'
                if CVf == True:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    fname = 'VF_'+fname
                    savePath = savePath + '_VF'
                if CRn == True:
                    cnt = int(self.RnSE.text())
                    fname = self.RnPE.text() +'_'+ str(cnt) +'.jpg'
                    cnt+=1
                    self.RnSE.setText(str(cnt))
                    savePath = savePath + '_RN'
                if not(os.path.isdir(savePath)):
                    os.makedirs(os.path.join(savePath))
                img.save(savePath+'/'+fname)
                #img.save(savePath+fname)

        self.RnSE.setText('0')

        #print(CRs, CRt, CHf, CVf, CRn)

    def createButton(self, text, member):
        button = Button(text)
        button.clicked.connect(member)
        return button
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
