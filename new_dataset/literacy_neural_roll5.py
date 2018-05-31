import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('C:\Users\RIJUL KATIYAR\Downloads\elementary_2015_16.csv')
ef=df.iloc[:,[4,5,8,11]]
ef.fillna(0)
ef=ef.rename(columns={'PERCENTAGE URBAN POPULATION': 'urban'})
ef=ef.rename(columns={'PERCENT URBAN POPULATION': 'urban','SEX RATIO' :'ratio'})
ef=ef.rename(columns={'TOTAL POULATION' : 'total'})
ef=ef[ef.urban>0]
ef=ef[ef.ratio>0]
ef=ef[ef.total>0]



ff=ef.values
temp=ff.T 
ff=ff.T
ratio=np.array(ff[2])
literacy=np.array(ff[3])
litmax=np.max(literacy)
arri=(literacy>=litmax)



total=np.array(ff[0])
urban=np.array(ff[1])
ratiomax=ratio[arri]
urbanmax=urban[arri]
print(ratiomax)
print(urbanmax)

tempin=temp[0:3]

tempout=temp[3:4]
max=np.amax(tempin)
tempin[1]=tempin[1]*max*0.01
tempin[2]=tempin[2]*max*0.001



tempin=tempin.T  #actual input
tempout=tempout.T #actual output
trainin=tempin[0:300]
trainout=tempout[300:600]
testin=tempin[0:300]
testout=tempout[300:600]







plt.plot(ratio,literacy)
plt.ylabel('literacy')
plt.xlabel('ratio')
plt.savefig('ratio_literacy')

plt.clf()
plt.plot(total,literacy)
plt.ylabel('literacy')
plt.xlabel('total')
plt.savefig('total_literacy')
plt.clf()
plt.plot(urban,literacy)
plt.ylabel('literacy')
plt.xlabel('urban')
plt.savefig('urban_literacy')
plt.clf()
plt.scatter(ratio,literacy)
plt.ylabel('literacy')
plt.xlabel('ratio')
plt.savefig('scatter_ratio_literacy')
plt.ylabel('literacy')
plt.xlabel('ratio')
plt.clf()
plt.scatter(total,literacy)
plt.ylabel('literacy')
plt.xlabel('total')
plt.savefig('scatter_total_literacy')
plt.ylabel('literacy')
plt.xlabel('total')
plt.clf()
plt.scatter(urban,literacy)
plt.savefig('scatter_urban_literacy')
plt.ylabel('literacy')
plt.xlabel('urban')
plt.clf()
label1=['female','male']
size1=[ratiomax,1000]
label2=['urban','rural']
size2=[urbanmax,100-urbanmax]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
fig1, ax1 = plt.subplots()
ax1.pie(size1,  labels=label1, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  
plt.tight_layout()
plt.savefig('ratiomax_pie')
plt.clf()
fig2, ax2 = plt.subplots()
ax2.pie(size2,  labels=label2, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  
plt.tight_layout()
plt.savefig('urbanmax_pie')
plt.clf()

def scale_size(data, data_min=None, data_max=None, size_min=10, size_max=60):

    # if the data limits are set to None we will just infer them from the data
    if data_min is None:
        data_min = data.min()
    if data_max is None:
        data_max = data.max()

    size_range = size_max - size_min
    data_range = data_max - data_min

    return ((data - data_min) *  size_range / data_range) + size_min
plt.figure(figsize=(1,1))


plt.scatter(urban, literacy, c=ratio, s=scale_size(total))
plt.colorbar()
plt.xlabel('urban')
plt.ylabel('literacy')

plt.show()


        



X=trainin
X=trainin/max
y=trainout/100


class Neural_Network(object):
  def __init__(self):

    self.inputSize = 3
    self.outputSize = 1
    self.hiddenSize = 10


    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 

  def predict(self):
    l1 = 1/(1 + np.exp(-(np.dot(X, NN.W1))))
    l2 = 1/(1 + np.exp(-(np.dot(l1, NN.W2))))


NN = Neural_Network()

learning_rate = .2 # slowly update the network
for epoch in range(10000):
    l1 = 1/(1 + np.exp(-(np.dot(X, NN.W1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, NN.W2))))
    er = (abs(y - l2)).mean()
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(NN.W2.T) * (l1 * (1-l1))
    NN.W2 += l1.T.dot(l2_delta) * learning_rate
    NN.W1 += X.T.dot(l1_delta) * learning_rate
    
NN.predict()

X=testin/max
y=testout/100
l1 = 1/(1 + np.exp(-(np.dot(X, NN.W1)))) # sigmoid function
l2 = 1/(1 + np.exp(-(np.dot(l1, NN.W2))))
err=(abs(l2-y))
plt.hist(err)
plt.xlabel('error')
plt.ylabel('frequency')
plt.savefig('hist')
plt.show()




accuracy=1-err.mean()
print(accuracy)




