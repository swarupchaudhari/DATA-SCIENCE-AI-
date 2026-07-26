import matplotlib.pyplot as plt 

#DATA
labels = ['Pyhton','Java','C++0','Javascript']
sizes = [40,25,20,15]

#plot
plt.pie(sizes,labels=labels,autopct='%1.1f%%')

#Title
plt.title("Programmimg Language USage")

#show 
plt.show()

