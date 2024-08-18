from pandas import read_csv
import matplotlib.pyplot as plt
df = read_csv('cn.csv')

ax = plt.subplot()
ax.plot(df.index,df.cn)
ax.set_xlabel('Iterations')
ax.set_ylabel(r'$c_n$')
ax.minorticks_on()
ax.grid(visible=True,which='both',axis='both')
plt.show()