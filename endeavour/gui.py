from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from endeavour import endeavour_functions
import csv
import os


# Location of csv file to save result to.
log_filename = r'results_log.csv'


master = Tk()
master.title("Machine Learning for Image Recognition Purposes")
master.geometry("450x500")
global source_filename
source_filename = None


softmax_dir = '..\mnist_softmax_models\softmax_alpha=0.1/'
softmax_name = 'model-999999.meta'

mlp_dir = '..\mnist_mlp_models_regularised\mlp_nlayers=1_nunits=200/'
mlp_name = 'model.meta'

conv_dir = '..\mnist_conv_models\conv_nlayers=1_nfeatures=32/'
conv_name = 'model.meta'



def load():
    global source_filename
    source_filename = askopenfilename()
    image = Image.open(source_filename).resize((200,200), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    lab_source.configure(image=photo)
    lab_source.image = photo


def convert():
    threshold = slide_threshold.get()
    endeavour_functions.process_image(source_filename, 'processed.jpg', threshold=threshold)
    image = Image.open('processed.jpg').resize((200,200))
    photo = ImageTk.PhotoImage(image)
    lab_dest.configure(image=photo)
    lab_dest.image = photo
    #lab_dest.pack()

def classify():
    if var.get() == 'Softmax Classifier':
        model_dir = softmax_dir
        model_name = softmax_name
    elif var.get() == 'MLP Classifier':
        model_dir = mlp_dir
        model_name = mlp_name
    else:
        model_dir = conv_dir
        model_name = conv_name

    print(var.get())
    print("Using classifier ", model_dir+model_name)

    digit = endeavour_functions.classify_image('processed.jpg', model_dir, model_name)
    ent_class.delete(0, END)
    ent_class.insert(0, str(digit[0]))
    
    ent_dist_0.delete(0, END)
    ent_dist_1.delete(0, END)
    ent_dist_2.delete(0, END)
    ent_dist_3.delete(0, END)
    ent_dist_4.delete(0, END)
    ent_dist_5.delete(0, END)
    ent_dist_6.delete(0, END)
    ent_dist_7.delete(0, END)
    ent_dist_8.delete(0, END)
    ent_dist_9.delete(0, END)

    ent_dist_0.insert(0, '{0:.6f}'.format(digit[1][0]))
    ent_dist_1.insert(0, '{0:.6f}'.format(digit[1][1]))
    ent_dist_2.insert(0, '{0:.6f}'.format(digit[1][2]))
    ent_dist_3.insert(0, '{0:.6f}'.format(digit[1][3]))
    ent_dist_4.insert(0, '{0:.6f}'.format(digit[1][4]))
    ent_dist_5.insert(0, '{0:.6f}'.format(digit[1][5]))
    ent_dist_6.insert(0, '{0:.6f}'.format(digit[1][6]))
    ent_dist_7.insert(0, '{0:.6f}'.format(digit[1][7]))
    ent_dist_8.insert(0, '{0:.6f}'.format(digit[1][8]))
    ent_dist_9.insert(0, '{0:.6f}'.format(digit[1][9]))


def save():
    classification = ent_class.get()
    probs = [ent_dist_0.get(), ent_dist_1.get(), ent_dist_2.get(), ent_dist_3.get(), ent_dist_4.get(),
             ent_dist_5.get(), ent_dist_6.get(), ent_dist_7.get(), ent_dist_8.get(), ent_dist_9.get()]
    classifier = var.get()
    input_dir, input_file = os.path.split(source_filename)
    input_par_dir = os.path.split(input_dir)[1]
    fields = [classification, *probs, classifier, input_file, input_par_dir, input_dir]
    with open(log_filename, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(fields)
    print('Saved results to log')





# Place all image operations in left grid
left_frame = Frame(master)
left_frame.pack(side=LEFT)
btn_load = Button(left_frame, text="Load Image", command=load)
btn_load.grid(row=0)
lab_source = Label(left_frame)
lab_source.grid(row=1)
thresh_frame = Frame(left_frame)
thresh_frame.grid(row=2)
btn_convert = Button(left_frame, text="Convert Image", command=convert)
btn_convert.grid(row=3)
lab_dest = Label(left_frame)
lab_dest.grid(row=4)

# Threshold label and slider in thresh frame
lab_thresh = Label(thresh_frame, text="Threshold:")
lab_thresh.pack(side=LEFT)
thresh_val = IntVar()
slide_threshold = Scale(thresh_frame, from_=0, to=255, variable=thresh_val, showvalue=0,
                        orient=HORIZONTAL, resolution=5)
slide_threshold.set(150)
slide_threshold.pack(side=LEFT)
lab_thresh_val = Label(thresh_frame, textvariable=thresh_val)
lab_thresh_val.pack(side=LEFT)

# Place all classification operations in right grid
right_frame = Frame(master)
right_frame.pack(side=RIGHT)
model_frame = Frame(right_frame)
model_frame.grid(row=0)
btn_classify = Button(right_frame, text="Classify Image", command=classify)
btn_classify.grid(row=1)
class_frame = Frame(right_frame)
class_frame.grid(row=2, pady=10)
lab_prob = Label(right_frame, text="Output Probability Distribution:")
lab_prob.grid(row=3)
dist_frame = Frame(right_frame)
dist_frame.grid(row=4)
btn_save = Button(right_frame, text="Save Results", command=save)
btn_save.grid(row=5, pady=10)

# Model label and selector in model frame
lab_model = Label(model_frame, text="Model:")
lab_model.pack(side=LEFT)
var = StringVar(master)
var.set('Softmax Classifier')
opt_model = OptionMenu(model_frame, var, 'Softmax Classifier', 'MLP Classifier', 'Convolutional Classifier')
opt_model.pack(side=LEFT)

# Classification label and value in class frame
lab_class = Label(class_frame, text="Predicted digit:")
lab_class.pack(side=LEFT)
ent_class = Entry(class_frame, width=5)
ent_class.pack(side=LEFT)

# Distribution frame
lab_dist_0 = Label(dist_frame, text="0:").grid(row=0)
lab_dist_1 = Label(dist_frame, text="1:").grid(row=1)
lab_dist_2 = Label(dist_frame, text="2:").grid(row=2)
lab_dist_3 = Label(dist_frame, text="3:").grid(row=3)
lab_dist_4 = Label(dist_frame, text="4:").grid(row=4)
lab_dist_5 = Label(dist_frame, text="5:").grid(row=0, column=2)
lab_dist_6 = Label(dist_frame, text="6:").grid(row=1, column=2)
lab_dist_7 = Label(dist_frame, text="7:").grid(row=2, column=2)
lab_dist_8 = Label(dist_frame, text="8:").grid(row=3, column=2)
lab_dist_9 = Label(dist_frame, text="9:").grid(row=4, column=2)

ent_dist_0 = Entry(dist_frame, width=8)
ent_dist_1 = Entry(dist_frame, width=8)
ent_dist_2 = Entry(dist_frame, width=8)
ent_dist_3 = Entry(dist_frame, width=8)
ent_dist_4 = Entry(dist_frame, width=8)
ent_dist_5 = Entry(dist_frame, width=8)
ent_dist_6 = Entry(dist_frame, width=8)
ent_dist_7 = Entry(dist_frame, width=8)
ent_dist_8 = Entry(dist_frame, width=8)
ent_dist_9 = Entry(dist_frame, width=8)

ent_dist_0.grid(row=0, column=1)
ent_dist_1.grid(row=1, column=1)
ent_dist_2.grid(row=2, column=1)
ent_dist_3.grid(row=3, column=1)
ent_dist_4.grid(row=4, column=1)
ent_dist_5.grid(row=0, column=3)
ent_dist_6.grid(row=1, column=3)
ent_dist_7.grid(row=2, column=3)
ent_dist_8.grid(row=3, column=3)
ent_dist_9.grid(row=4, column=3)

"""
opt_model.pack()
slide_threshold.pack()
btn_load.pack()
btn_convert.pack()
btn_classify.pack()
lab_source.pack()
lab_dest.pack()
lab_class.pack()
lab_prob.pack()
"""

mainloop()