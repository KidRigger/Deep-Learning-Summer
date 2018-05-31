import numpy as np
input=np.array([[0,0,1],[1,0,1],[0,0,0],[0,1,0]])
output=np.array([[1],[0],[0],[1]])
count=0
lr=.1
def sigmoid(s) :
			return (1/(1+np.exp(-s)))
def derivative(s) :
			return s*(1-s)
w1=np.random.rand(3,2)
w2=np.random.rand(2,1)
while count<2000 :
	

	hin=np.dot(input,w1)
	hin=sigmoid(hin)
	nout=np.dot(hin,w2)
	nout=sigmoid(nout)
	error=output-nout
	slopeop=derivative(nout)
	slopehid=derivative(hin)
	dop=error*slopeop
	errorhid=np.dot(dop,w2.T)
	dhid=errorhid*slopehid
	w21=np.dot(hin.T,dop)
	w21=w21*lr
	w2=np.add(w2,w21)
	w1=np.add(w1,np.dot(input.T,dhid)*lr)
	count=count+1
	
cnt=0
while cnt<10 :
	test=int(raw_input("enter a number"))
	arr=np.array([0,0,0])
	arr[2]=test%10
	test=test/10
	arr[1]=test%10
	test=test/10
	arr[0]=test%10
	hin=np.dot(arr,w1)
	hin=sigmoid(hin)
	nout=np.dot(hin,w2)
	nout=sigmoid(nout)
	print(arr)
	print(nout)
	cnt=cnt+1
	
	
	
	
	