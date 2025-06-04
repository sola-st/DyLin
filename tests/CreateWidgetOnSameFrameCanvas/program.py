from nltk.draw.util import CanvasFrame, CanvasWidget, SequenceWidget
from tkinter import Tk

class CircleWidget(CanvasWidget):
    def __init__(self, canvas, x, y, radius):
        self._circle_id = canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="blue")
        super().__init__(canvas)
    
    def _tags(self):
        return (str(self._circle_id),)

class SimpleTextWidget(CanvasWidget):
    def __init__(self, canvas, text):
        self._text_id = canvas.create_text(10, 10, anchor="nw", text=text, fill="black")
        super().__init__(canvas)
    
    def __repr__(self):
        return "<SimpleTextWidget>"
    
    def _tags(self):
        return (str(self._text_id),)


# ====== OK ========
root = Tk()
cf = CanvasFrame(root, width=400, height=200)
canvas = cf.canvas()

text_widget = SimpleTextWidget(canvas, "Hello, NLTK Canvas!")
circle_widget = CircleWidget(canvas, 100, 100, 20)

sequence_widget = SequenceWidget(canvas, text_widget, circle_widget, align='center', space=10)

cf.add_widget(sequence_widget)


# ====== NOT OK 1 ========
root = Tk()
cf_1 = CanvasFrame(root, width=400, height=200)
cf_2 = CanvasFrame(root, width=200, height=400)
canvas_1 = cf_1.canvas()
canvas_2 = cf_2.canvas()

text_widget = SimpleTextWidget(canvas_1, "Hello, NLTK Canvas!")
circle_widget = CircleWidget(canvas_2, 100, 100, 20)

sequence_widget = SequenceWidget(canvas_2, text_widget, circle_widget, align='center', space=10)

cf_1.add_widget(sequence_widget) # DyLin warn

# ====== NOT OK 2 ========
root = Tk()
cf_1 = CanvasFrame(root, width=400, height=200)
cf_2 = CanvasFrame(root, width=200, height=400)
canvas_1 = cf_1.canvas()
canvas_2 = cf_2.canvas()

circle_widget_1 = CircleWidget(canvas_1, 100, 100, 20)
circle_widget_2 = CircleWidget(canvas_2, 100, 100, 20)

cf_1.add_widget(canvaswidget=circle_widget_1)
cf_1.add_widget(canvaswidget=circle_widget_2) # DyLin warn

# ====== NOT OK 3 ========
root = Tk()
cf_1 = CanvasFrame(root, width=400, height=200)
cf_2 = CanvasFrame(root, width=200, height=400)
canvas_1 = cf_1.canvas()
canvas_2 = cf_2.canvas()

circle_widget_2 = CircleWidget(canvas_2, 100, 100, 20)

# this will throw an error because there isn't at least one canvasWidget on the canvas of CanvasFrame
cf_1.add_widget(canvaswidget=circle_widget_2) # DyLin warn
