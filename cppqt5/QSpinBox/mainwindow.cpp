#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QTimer>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //====================================
    // 추가할 부분 시작
    QTimer *timer = new QTimer(this);
    // 0.1초마다 타이머가 걸린다.
    timer->setInterval(100);
    connect(timer, &QTimer::timeout, this, &MainWindow::onTimeOut);
    timer->start();
    // 추가할 부분 끝
    //====================================
}

MainWindow::~MainWindow()
{
    delete ui;
}

//====================================
// 추가할 부분 시작
void MainWindow::onTimeOut()
{
    //value가 100보다 작거나 같을 때 까지 value는 1씩 증가한다.
    int value = ui->progressBar->value();
    if(value>=100) {
        return;
    }
    value+=1;
    ui->progressBar->setValue(value);
}
// 추가할 부분 끝
//====================================
