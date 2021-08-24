from application.lib.cam import Camera
import cv2
import tkinter as tk
from tkinter import font
from application.lib.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
from application.lib.processing import findFaceGetPulse
from application.lib.emotions import Emotions
from PIL import Image, ImageTk
import numpy as np
import datetime
from application.text_strings import *


class Pulse(object):

	def __init__(self):
		self.camera = Camera(camera=0)
		if not self.camera.valid:
			raise ValueError("ERROR!")
		self.w, self.h = 0, 0
		self.e = Emotions()
		self.processor = findFaceGetPulse(emotions=self.e)
		self.logged_in = False
		self.bpm = 0
		self.sending_pulse = False
		self.sending_emotions = False
		self.bpm_plot = False
		self.plot_title = "Data display - raw signal (top) and Power Spectral Density (bottom)"
	def start(self):
		self.processor.find_faces_toggle()

	def toggle_search(self):

		state = self.processor.find_faces_toggle()
		print("face detection lock =", not state)

	def make_bpm_plot(self):

		plotXY([[self.processor.times,
				 self.processor.samples],
				[self.processor.freqs,
				 self.processor.fft]],
			   labels=[False, True],
			   showmax=[False, "bpm"],
			   label_ndigits=[0, 0],
			   showmax_digits=[0, 1],
			   skip=[3, 3],
			   name=self.plot_title,
			   bg=self.processor.slices[0])

	def toggle_display_plot(self):
		"""
        Toggles the data display.
        """
		if self.bpm_plot:
			print("bpm plot disabled")
			self.bpm_plot = False
			destroyWindow(self.plot_title)
		else:
			print("bpm plot enabled")
			if self.processor.find_faces:
				self.toggle_search()
			self.bpm_plot = True
			self.make_bpm_plot()
			moveWindow(self.plot_title, self.w, 0)
			self.processor.plot()

	def mean(self, x):
		length = len(x)
		if length is 0:
			return "---"

		sum = 0
		for a in x:
			sum = sum + a
		return int(sum / length)

	def loop(self):
		frame = self.camera.get_frame()
		frame = cv2.flip(frame, 1)
		self.h, self.w, _c = frame.shape
		self.processor.frame_in = frame
		# process the image frame to perform all needed analysis
		self.processor.run()
		# collect the output frame for display
		output_frame = self.processor.frame_out

		output_frame = cv2.resize(output_frame, (800, 452))
		cv2image = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGBA)
		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)
		lmain.imgtk = imgtk
		lmain.configure(image=imgtk)

		bpms = self.processor.get_bpms()
		self.bpm = self.mean(bpms)

		if self.processor.gap:
			text_var_pulse.set(TXT_WAIT + str(int(self.processor.gap)) + " s")
		else:
			text_var_pulse.set(str(self.bpm))
		if self.processor.find_faces is True:
			text_var_pulse.set("---")

		set_emotions_labels(self.e.get_last_prediction())
		if self.bpm_plot:
			self.make_bpm_plot()


def set_emotions_labels(emotions):
	for i in range(7):
		f = font.Font(label_pulse, label_pulse.cget("font"))
		labels_emotion[i].configure(font=f, fg='black')
		labels_emotions_value[i].configure(font=f, fg='black')
		e = emotions[i]
		e = "{:0.2f}%".format(e * 100)
		var_emotions[i].set(e)

	max_emotion = emotions.argmax()
	bold_font = font.Font(labels_emotion[max_emotion], labels_emotion[max_emotion].cget("font"))
	bold_font.configure(weight="bold")
	labels_emotion[max_emotion].configure(font=bold_font, fg='red')
	labels_emotions_value[max_emotion].configure(font=bold_font, fg='red')



p = Pulse()
root = tk.Tk()
root.title(TXT_TITLE)
'''
def write_csv():
	"""
    Writes current data to a csv file
    """
	fn = "Webcam-pulse" + str(datetime.datetime.now())
	fn = fn.replace(":", "_").replace(".", "_")
	data = np.vstack((p.processor.times, p.processor.samples)).T
	np.savetxt(fn + ".csv", data, delimiter=',')
	print("Writing csv")
'''
def start_pulse_measure():
	p.start()
	button_start_measure.config(text=TXT_STOP_MEASURE_BUTTON)
	button_start_measure.config(command=stop_pulse_measure)


def stop_pulse_measure():
	p.start()
	button_start_measure.config(text=TXT_START_MEASURE_BUTTON)
	button_start_measure.config(command=start_pulse_measure)
	text_var_pulse.set("---")


def on_enter(event):
	event.widget.config(fg="blue")


def on_leave(event):
	event.widget.config(fg="black")


# ----------------- FRAME PULSE ---------------------

frame_pulse = tk.Frame(root, highlightbackground="black", highlightcolor="black", highlightthickness=1, bd=0)
frame_pulse.grid(column=0, row=5, rowspan=5, sticky=tk.E + tk.W + tk.N + tk.S, pady=3, padx=3)
frame_pulse.lift()

# PULSE label
label_pulse = tk.Label(frame_pulse, text=TXT_PULSE, width=21)
label_pulse.grid(column=0, row=0, sticky=tk.E + tk.W + tk.N + tk.S)
bold_font = font.Font(label_pulse, label_pulse.cget("font"))
bold_font.configure(weight="bold")
bold_font.configure(size=18)
label_pulse.configure(font=bold_font)

# pulse result
text_var_pulse = tk.StringVar()
text_var_pulse.set("---")
label_pulse_result = tk.Label(frame_pulse, textvariable=text_var_pulse, fg="red")
label_pulse_result.config(font=("Courier", 20))
label_pulse_result.grid(column=0, row=1, sticky=tk.E + tk.W + tk.N + tk.S)

# measure start button
button_start_measure = tk.Button(frame_pulse, text=TXT_START_MEASURE_BUTTON, command=start_pulse_measure)
button_start_measure.grid(column=0, row=2)

button_plot = tk.Button(frame_pulse, text=TXT_PLOT, command=p.toggle_display_plot)
button_plot.grid(column=2, row=2)
'''
button_csv = tk.Button(frame_pulse, text=TXT_CSV, command=write_csv)
button_csv.grid(column=4, row=2)
'''
# ----------------- FRAME PULSE END -----------------


# ----------------- FRAME EMOTIONS ------------------

def insert_row(frame=None, index=None, text=None, text_var=None):
	l1 = tk.Label(frame, borderwidth=1, relief="groove", width=22, text=text)
	l1.grid(column=0, row=index, sticky=tk.E + tk.W + tk.N + tk.S)
	l2 = tk.Label(frame, borderwidth=1, relief="groove", width=10, textvariable=text_var)
	l2.grid(column=1, row=index, sticky=tk.E + tk.W + tk.N + tk.S)
	return l1, l2


frame_emotions = tk.Frame(root, bd=0)
frame_emotions.grid(column=0, row=10, rowspan=7, sticky=tk.E + tk.W + tk.N + tk.S, pady=3, padx=3)
frame_emotions.lift()

labels_emotion = [None] * 7
labels_emotions_value = [None] * 7
var_emotions = [tk.StringVar() for i in range(7)]

emotions_l = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Indifference']

# row for each emotion
for i in range(7):
	var_emotions[i].set("")
	(labels_emotion[i], labels_emotions_value[i]) = insert_row(frame=frame_emotions, index=i, text=emotions_l[i],
															   text_var=var_emotions[i])

# ----------------- FRAME EMOTIONS END --------------


# ----------------- FRAME MAIN VIDEO ----------------

# main video stream
lmain = tk.Label(root)
lmain.grid(column=1, row=0, columnspan=4, rowspan=17)
lmain.lower()

# ----------------- FRAME MAIN VIDEO END ------------
while True:
	p.loop()
	root.update_idletasks()
	root.update()
