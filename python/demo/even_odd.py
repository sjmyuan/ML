from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD,Adagrad
from keras.callbacks import EarlyStopping
import numpy
import keras.initializations
import matplotlib.pyplot as plt

#构建模型    
model = Sequential()
model.add(Dense(output_dim=32, input_dim=16,init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=2))
model.add(Activation('sigmoid'))


#编译模型
adagrad = Adagrad(lr=0.3, epsilon=1e-06)
model.compile(loss='mean_squared_error',
              optimizer=adagrad)

#准备数据
def to_binary(num):
    return map(int,'{0:016b}'.format(num))

def to_tuple(i):
    return [i%2,(i+1)%2] 

data=range(1,20000,1)
expect=map(to_tuple,data)

xvalue=numpy.asarray(map(to_binary,data))
yvalue=numpy.asarray(expect)

print xvalue.shape

#训练模型
info=model.fit(xvalue, yvalue,
          nb_epoch=100,
          batch_size=16,
          validation_split=0.4,
          shuffle=True,
          callbacks=[EarlyStopping(patience=10,verbose=1)],
          show_accuracy=True)
          
#预测值
vals=model.predict(numpy.asarray([to_binary(3)]))

print vals
