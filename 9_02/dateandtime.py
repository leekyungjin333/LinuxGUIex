from PyQt5.QtCore import QDate, QTime, Qt
now = QDate.currentDate()
time = QTime.currentTime()
print(now.toString('yyyy-MM-dd'))
print(time.toString('hh:mm:ss'))

