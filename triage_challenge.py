import numpy as np
import math

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall

import sys
import json

#--------------------------------------------------------------------------------------------------------
# GLOBAL SETTINGS
#--------------------------------------------------------------------------------------------------------

FFT_NUM_BARS                            = 200
REGRESSION_DEGREE                       = 5
MAX_VAL_NORMALIZE                       = 200.0

def fill_missing_values_with_polynomial_regression(data):

    """
    Generates a polynomial of degree N using non-None values from the data.
    Then, creates a new vector with original non-None values and predicted values for None entries.

    :param data: List of numbers with some None values
    :param N: Degree of the polynomial to fit
    :return: New vector with predicted values in place of None
    """
    # Extracting (x, y) pairs where y is not None
    x_vals = [i for i, y in enumerate(data) if y is not None]
    y_vals = [y for y in data if y is not None]

    # Fitting a polynomial of degree N
    coefficients = np.polyfit(x_vals, y_vals, REGRESSION_DEGREE)
    polynomial = np.poly1d(coefficients)

    # Generating the new vector
    new_vector = [polynomial(x) if y is None else y for x, y in enumerate(data)]
    return new_vector
def convert_line(line,data_type ):
    global MAX_VAL_NORMALIZE
    max_value = 0.0
    cols    = line.replace("\n","").split(",")
    input   = []
    output  = []
    subject = 0
    for i in range(1,337):
        if cols[i] == "":
            input.append(None)
        else:
            input.append( float(cols[i] )/ MAX_VAL_NORMALIZE)
    if data_type == "train":
        output                  = [0.0, 0.0, 0.0 ,0.0 ,0.0, 0.0]
        output[int(cols[-1])]   = 1.0
    subject = int(cols[0])
    return subject, input,output
def load_data( data_type="train",fix_type="polynomial",use_wave=True,user_fft=True):
    print("-"*80)
    aorta_lines = open("data/aortaP_"+data_type+"_data.csv").readlines()
    brach_lines = open("data/brachP_"+data_type+"_data.csv").readlines()
    del aorta_lines[0]
    del brach_lines[0]
    inputs  = []
    outputs = []
    subject = 0
    for aorta_line,brach_line  in zip(aorta_lines,brach_lines):
        if len(aorta_line.split(","))>=337 and len(brach_line.split(","))>=337:
            inp, aorta,fixed_aorta,aorta_fft,brach,fixed_brach,brach_fft,out = [], [],[],[],[],[],[],[]
        
            if data_type =="train":
                subject, aorta,out       = convert_line( aorta_line,data_type)
                subject, brach,_         = convert_line(aorta_line,data_type)
            else:
                subject, aorta,_ = convert_line(aorta_line,data_type)
                subject, brach,_ = convert_line(brach_line,data_type)
                out     = []

            if fix_type=="polynomial":
                fixed_aorta = fill_missing_values_with_polynomial_regression(aorta)
                fixed_brach = fill_missing_values_with_polynomial_regression(brach)

            if use_wave:
                for a in fixed_aorta:
                    inp.append(a)
                for b in fixed_brach:
                    inp.append(b)


            sys.stdout.write("\rLoading... {} subjects".format(subject))
            sys.stdout.flush()

            inputs.append(inp)
            outputs.append(out)

    x_train, x_val, y_train, y_val = inputs,[], outputs,[] 
    sys.stdout.write("\rLoading... {} subjects ... DONE!".format(subject))
    sys.stdout.write("\Saving data... {} subjects ... DONE!".format(subject))

    print("Converting data to numpy matrix...")
    x_train                             = np.array(x_train)
    y_train                             = np.array(y_train)
    x_val                               = np.array(x_val)
    y_val                               = np.array(y_val)

    return x_train, x_val, y_train, y_val
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        try:
            return 2 * ((precision * recall) / (precision + recall))
        except:
            return 0
def CONFIG_TENSORFLOW():
    print("-"*80)
    print("Tensorflow configuration...",end="")
    num_cpu_threads = len(tf.config.list_physical_devices('CPU')) * 16
    tf.config.threading.set_intra_op_parallelism_threads(num_cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_cpu_threads)
    print("Tensorflow num_cpu_threads: ",num_cpu_threads)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                print("Tensorflow GPU: ",gpu)
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    tf.config.run_functions_eagerly(True)
    print("DONE")
def GENERATE_OUTPUT_JSON(model,data_type,x_data,format=False):
    json_output = {}
    y_pred = model.predict(x_data)
    n_class = len(y_pred[0])
    subject = 0
    for pred in y_pred.tolist():
        m = 0
        c = 0
        for i in range(0,n_class):
            if m <pred[i]:
                m = pred[i]
                c = i
        json_output[subject] = c
        subject+=1
    if format:
        open("MGB-Harvard_output.json","w").write( str(json.dumps(json_output, indent=1)).replace('"',"")) 
    else:
        open("MGB-Harvard_output.json","w").write( str(json_output )) 
    return json_output

#--------------------------------------------------------------------------------------------------------
# MODEL FUNCTIONS
#--------------------------------------------------------------------------------------------------------

#start tensorflow settings
CONFIG_TENSORFLOW()

DATA_TYPE   = "test"          

#load the data
input_data, _, _, _ = load_data(data_type=DATA_TYPE,fix_type="polynomial",use_wave=True,user_fft=False)

#load the trained model
model = load_model("models/model_0000169.h5",custom_objects={'F1Score': F1Score})

#generate the output json file in the requested format
GENERATE_OUTPUT_JSON(model,DATA_TYPE,input_data,format=True)
