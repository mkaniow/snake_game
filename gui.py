import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLineEdit, QLabel, QRadioButton, QTableWidget, QTableWidgetItem, QMessageBox, QSpinBox
import game
import agent_ql
import agent_dql

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'SNAKE'
        self.left = 500
        self.top = 500
        self.width = 1024
        self.height = 500
        self.setStyleSheet("background-image: url(unnamed.jpg)")
        self.initUI()
        
    def create_label(self, name, x, y):
        label = QLabel(name, self) 
        label.move(x, y)
        label.setStyleSheet(
            "color: white;"
        )
        return label

    def create_line_edit(self, x, y, height, width):
        line = QLineEdit(self) 
        line.move(x, y)
        line.resize(height, width)
        line.setStyleSheet(
            "background: #FFFFFF;" +
            "border: 1px solid '#008080';" +
            "border-radius: 10px;"
            )
        return line
    
    def create_spinbox(self, x, y, min, max, height, width):
        spin_box = QSpinBox(self)
        spin_box.move(x, y)
        spin_box.resize(height, width)
        spin_box.setMinimum(min)
        spin_box.setMaximum(max)
        return spin_box
    
    def create_nn_spinbox(self, x, y, min, max, height, width):
        spin_box = QSpinBox(self)
        spin_box.move(x, y)
        spin_box.resize(height, width)
        spin_box.setMinimum(min)
        spin_box.setMaximum(max)
        spin_box.valueChanged.connect(self.aha)
        return spin_box

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.rb1 = QRadioButton("Play game", self, clicked= lambda: self.create_table(self.rb1))
        self.rb1.setGeometry(15, 10, 100, 30)
        self.rb1.setStyleSheet(
            "color: white;"
        )
  
        self.rb2 = QRadioButton("Create Q-learning agent", self, clicked= lambda: self.create_table(self.rb2))
        self.rb2.setGeometry(15, 50, 100, 30)
        self.rb2.setStyleSheet(
            "color: white;"
        )

        self.rb3 = QRadioButton("Create deep Q-learning agent", self, clicked= lambda: self.create_table(self.rb3))
        self.rb3.setGeometry(15, 90, 100, 30)
        self.rb3.setStyleSheet(
            "color: white;"
        )

        self.game_speed = self.create_label('Game speed [10, 30]', 200, 10)
        self.game_speed.hide()
        self.speed_box = self.create_spinbox(200, 40, 10, 30, 50, 25)
        self.speed_box.hide()
        
        self.label_games = self.create_label("Training games 1+", 10, 180)
        self.label_games.hide()
        self.games = self.create_line_edit(10, 210, 50, 20)
        self.games.hide()

        self.label_GAMMA = self.create_label("Gamma [0, 1]", 10, 240)
        self.label_GAMMA.hide()
        self.GAMMA = self.create_line_edit(10, 270, 50, 20)
        self.GAMMA.hide()
        
        self.label_LR = self.create_label("Learning Rate [0, 1]", 10, 300)
        self.label_LR.hide()
        self.LR = self.create_line_edit(10, 330, 50, 20)
        self.LR.hide()

        self.label_MS = self.create_label("Memory size of NN", 900, 180)
        self.label_MS.hide()
        self.MS = self.create_line_edit(900, 210, 50, 20)
        self.MS.hide()

        self.label_BS = self.create_label("Batch size of NN", 900, 240)
        self.label_BS.hide()
        self.BS = self.create_line_edit(900, 270, 50, 20)
        self.BS.hide()

        self.label_nn_1 = self.create_label("Layer 1", 400, 10)
        self.label_nn_1.hide()
        self.nn_1 = self.create_line_edit(400, 40, 50, 20)
        self.nn_1.hide()

        self.label_nn_2 = self.create_label("Layer 2", 470, 10)
        self.label_nn_2.hide()
        self.nn_2 = self.create_line_edit(470, 40, 50, 20)
        self.nn_2.hide()

        self.label_nn_3 = self.create_label("Layer 3", 540, 10)
        self.label_nn_3.hide()
        self.nn_3 = self.create_line_edit(540, 40, 50, 20)
        self.nn_3.hide()

        self.label_nn_4 = self.create_label("Layer 4", 610, 10)
        self.label_nn_4.hide()
        self.nn_4 = self.create_line_edit(610, 40, 50, 20)
        self.nn_4.hide()

        self.label_nn_5 = self.create_label("Layer 5", 680, 10)
        self.label_nn_5.hide()
        self.nn_5 = self.create_line_edit(680, 40, 50, 20)
        self.nn_5.hide()

        self.label_nn_6 = self.create_label("Layer 6", 750, 10)
        self.label_nn_6.hide()
        self.nn_6 = self.create_line_edit(750, 40, 50, 20)
        self.nn_6.hide()

        self.label_number_layers = self.create_label('Layers [1, 5]', 200, 70)
        self.label_number_layers.hide()
        self.number_layers = self.create_nn_spinbox(200, 100, 1, 5, 50, 25)
        self.number_layers.hide()

        self.button = QPushButton('START', self)
        self.button.setGeometry(473, 450, 70, 30)
        self.button.setStyleSheet(
            "background: #FFFFFF;" +
            "border: 1px solid '#008080';" +
            "border-radius: 10px;"
            )
        self.button.clicked.connect(self.on_click)
        self.show()

    def hide_elements(self, *args):
        for arg in args:
            arg.hide()

    def show_elements(self, *args):
        for arg in args:
            arg.show()
    
    def aha(self):
        layers = self.number_layers.value()
        labels = [self.label_nn_1, self.label_nn_2, self.label_nn_3, self.label_nn_4, self.label_nn_5]
        boxes = [self.nn_1, self.nn_2, self.nn_3, self.nn_4, self.nn_5]
        for i in range(len(labels)):
            if i < layers:
                labels[i].show()
                boxes[i].show()
            else:
                labels[i].hide()
                boxes[i].hide()

    def create_table(self, name):
        if name.text() == "Play game":
            if name.isChecked():
                self.show_elements(self.speed_box, self.game_speed)
                self.hide_elements(self.label_LR, self.LR, self.label_GAMMA, self.GAMMA, self.label_games, self.games, self.label_MS, self.MS, self.label_BS, self.BS, self.label_nn_1, self.nn_1, self.label_nn_2, self.nn_2, self.label_nn_3, self.nn_3, self.label_nn_4, self.nn_4, self.label_nn_5, self.nn_5, self.label_number_layers, self.number_layers)
        if name.text() == "Create Q-learning agent":
            if name.isChecked():
                self.show_elements(self.speed_box, self.game_speed, self.label_LR, self.LR, self.label_GAMMA, self.GAMMA, self.label_games, self.games)
                self.hide_elements(self.label_MS, self.MS, self.label_BS, self.BS, self.label_nn_1, self.nn_1, self.label_nn_2, self.nn_2, self.label_nn_3, self.nn_3, self.label_nn_4, self.nn_4, self.label_nn_5, self.nn_5, self.label_number_layers, self.number_layers)
        if name.text() == "Create deep Q-learning agent":
            if name.isChecked():
                self.show_elements(self.speed_box, self.game_speed, self.label_LR, self.LR, self.label_GAMMA, self.GAMMA, self.label_games, self.games, self.label_MS, self.MS, self.label_BS, self.BS, self.label_nn_1, self.nn_1, self.label_number_layers, self.number_layers)
                self.hide_elements(self.label_nn_2, self.nn_2, self.label_nn_3, self.nn_3, self.label_nn_4, self.nn_4, self.label_nn_5, self.nn_5)
                self.aha()

    def warning(self):
        self.warning_box = QMessageBox()
        self.warning_box.setWindowTitle("WARNING!")
        self.warning_box.setText("Please check your inputs")
        self.warning_box.setIcon(QMessageBox.Warning)
        self.warning_box.exec_()

    def popup_box(self):
        self.warning_box = QMessageBox()
        self.warning_box.setWindowTitle("Important question")
        self.warning_box.setText("Do you want to play again")
        self.warning_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        returnValue = self.warning_box.exec()
        if returnValue == QMessageBox.Ok:
            self.game.reset()
        else:
            self.game.quit_game()

    def on_click(self):
        if self.rb1.isChecked():
            try:
                speed = self.speed_box.value()
                print('1')
                self.game = game.SnakeGame(speed, human=True)
                print('2')
                #game loop
                while True:
                    game_over, score, reward = self.game.play_step(self.game.direction)

                    if game_over == True:
                        self.popup_box()
            except:
                print('ojojojo')
        elif self.rb2.isChecked():
            try:
                speed = self.speed_box.value()
                LR = float(self.LR.text())
                if LR > 1:
                    LR = 0.99
                elif LR < 0:
                    LR = 0.01
                GAMMA = float(self.GAMMA.text())
                if GAMMA > 1:
                    GAMMA = 0.99
                elif GAMMA < 0:
                    GAMMA = 0.01
                training_games = int(self.games.text())
                if training_games < 1:
                    training_games = 1
                try:
                    self.agent_ql = agent_ql.train(LR, GAMMA, training_games, speed)
                except:
                    pass
            except:
                self.warning()   
        elif self.rb3.isChecked():
            try:
                speed = self.speed_box.value()
                GAMMA = float(self.GAMMA.text())
                if GAMMA > 1:
                    GAMMA = 0.99
                elif GAMMA < 0:
                    GAMMA = 0.01
                LR = float(self.LR.text())
                if LR > 1:
                    LR = 0.99
                elif LR < 0:
                    LR = 0.01
                training_games = int(self.games.text())
                if training_games < 1:
                    training_games = 1
                MS = int(self.MS.text())
                BS = int(self.BS.text())
                if self.number_layers.value() == 1:
                    try:
                        NN_1 = int(self.nn_1.text())
                        self.agent_dql = agent_dql.train(speed, LR, GAMMA, training_games, BS, 11, MS, NN_1)
                    except:
                        pass
                elif self.number_layers.value() == 2:
                    try:
                        NN_1 = int(self.nn_1.text())
                        NN_2 = int(self.nn_2.text())
                        self.agent_dql = agent_dql.train(speed, LR, GAMMA, training_games, BS, 11, MS, NN_1, NN_2)
                    except:
                        pass
                elif self.number_layers.value() == 3:
                    try:
                        NN_1 = int(self.nn_1.text())
                        NN_2 = int(self.nn_2.text())
                        NN_3 = int(self.nn_3.text())
                        self.agent_dql = agent_dql.train(speed, LR, GAMMA, training_games, BS, 11, MS, NN_1, NN_2, NN_3)
                    except:
                        pass
                elif self.number_layers.value() == 4:
                    try:
                        NN_1 = int(self.nn_1.text())
                        NN_2 = int(self.nn_2.text())
                        NN_3 = int(self.nn_3.text())
                        NN_4 = int(self.nn_4.text())
                        self.agent_dql = agent_dql.train(speed, LR, GAMMA, training_games, BS, 11, MS, NN_1, NN_2, NN_3, NN_4)
                    except:
                        pass
                elif self.number_layers.value() == 5:
                    try:
                        NN_1 = int(self.nn_1.text())
                        NN_2 = int(self.nn_2.text())
                        NN_3 = int(self.nn_3.text())
                        NN_4 = int(self.nn_4.text())
                        NN_5 = int(self.nn_5.text())
                        self.agent_dql = agent_dql.train(speed, LR, GAMMA, training_games, BS, 11, MS, NN_1, NN_2, NN_3, NN_4, NN_5)
                    except:
                        pass
            except:
                self.warning() 


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())