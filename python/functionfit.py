from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD,Adagrad
from keras.callbacks import EarlyStopping
import numpy
import keras.initializations
import matplotlib.pyplot as plt

def cstUniform(shape):
    return keras.initializations.uniform(shape,1)

#构建模型    
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1,init='uniform'))
model.add(Dense(output_dim=10,))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=1))


#编译模型
adagrad = Adagrad(lr=0.3, epsilon=1e-06)
model.compile(loss='mean_squared_error',
              optimizer=adagrad)

#准备数据
xvalue=numpy.asarray([numpy.arange(0,2,0.0002)]).T
yvalue=numpy.sin(xvalue*2*numpy.pi)*0.8
num=len(xvalue)
trainset= (numpy.array([xvalue[i] for i in range(0,num,3)]),numpy.array([yvalue[i] for i in range(0,num,3)]))
validset= (numpy.array([xvalue[i] for i in range(1,num,9)]),numpy.array([yvalue[i] for i in range(1,num,9)]))

#训练模型
info=model.fit(trainset[0], trainset[1],
          nb_epoch=400,
          batch_size=16,
          validation_data=validset,
          callbacks=[EarlyStopping(patience=10,verbose=1)],
          show_accuracy=True)
          
#绘制图表
r_val=model.predict_on_batch(xvalue)
fig,(ax1)=plt.subplots(1,1,True,True)       
ax1.plot(xvalue,r_val)                  
ax1.plot(xvalue.reshape((xvalue.size,)),yvalue.reshape((yvalue.size,)))                      
plt.show() 
