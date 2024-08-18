from pandas import read_csv
import matplotlib.pyplot as plt
from datetime import datetime
df = read_csv('cn.csv')

ax = plt.subplot()
ax.plot(df.index,df.cn)
ax.set_xlabel('Iterations')
ax.set_ylabel(r'$c_n$')
ax.minorticks_on()
ax.grid(visible=True,which='both',axis='both')
plt.show()
fn = datetime.now().strftime("%d%m%Y_%H%M")+'.png'
ax.figure.savefig(fn)