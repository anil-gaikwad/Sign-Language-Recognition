import cv2
from PIL import Image, ImageTk
import tkinter as tk
from keras.models import model_from_json
import operator
from string import ascii_uppercase


class Signsystem:
    def __init__(self):
        self.directory = "models/"
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # open json model
        self.json_file = open(self.directory + "model-bw.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights(self.directory + "model-bw.h5")
        ##
        #self.json_file_tkdi = open(self.directory + "model-bw_tkdi.json", "r")
        #self.model_json_tkdi = self.json_file_tkdi.read()
        #self.json_file_tkdi.close()

        #self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        #self.loaded_model_tkdi.load_weights(self.directory + "model-bw_tkdi.h5")

        ###
        self.ct = {'blank': 0}
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0

        # starting model loading
        print("model loading")
        self.root = tk.Tk()
        self.root.title("SRL")
        self.root.geometry("800x730")
        self.panel = tk.Label(self.root)
        self.panel.place(x=135, y=10, width=640, height=640)

        # initialize image  frame
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=460, y=95, width=310, height=310)
        self.T = tk.Label(self.root)
        self.T.place(x=171, y=17)
        self.T.config(text="Sign Language Recognition", fg="white", bg="#2B3856", font=("Arial", 30))

        # character panel
        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=600)
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=600)
        self.T1.config(text="Predict Char =>", fg="white", bg="brown", font=("Arial", 25))

        # word panel
        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=660)
        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=660)
        self.T2.config(text="Word =>", fg="white", bg="#8a2e7f", font=("Arial", 25))

        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            # creating frame
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol, font=("Arial", 30))
            self.panel4.config(text=self.word, font=("Arial", 30))

        self.root.after(30, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        #result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))

        prediction = dict()
        prediction['blank'] = result[0][0]
        index = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][index]
            index += 1
        # LAYER 1
        # Classify between 27 Symbols
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        # LAYER 2
        # Classify between Similar Symbols
        #if self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T':
           #prediction = {'D': result_tkdi[0][0],
           #              'I': result_tkdi[0][1],
          #               'K': result_tkdi[0][2],
         #                'T': result_tkdi[0][3]}

        #prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        #self.current_symbol = prediction[0][0]

        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    # for i in ascii_uppercase:
                    # self.ct[i] = 0
                    return
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol


if __name__ == '__main__':
    print("Starting System")
    ss = Signsystem()
    ss.root.mainloop()
    print("Close System")
