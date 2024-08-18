import os
from pandas import read_csv,DataFrame
import matplotlib.pyplot as plt
from datetime import datetime

appliedd_df = read_csv(
    os.sep.join([os.getcwd(),'appliedd.csv']),index_col=0
)

indices = appliedd_df[appliedd_df.appliedd>0].index

body_df = read_csv(
    os.sep.join([os.getcwd(),'body.csv']),index_col=0
)

volumes = body_df.iloc[indices,:]['volume']
data = []
for root,dirs,files in os.walk(os.getcwd()):
    for file in files:
        if file.upper().__contains__('OUTPUT') and file.upper().__contains__('CSV'):
            print(file)
            fp = os.sep.join([root,file])
            df = read_csv(fp)
            step = file.split('.')[0].split('_')[1]
            df = df.iloc[indices,:][['dispy','forcey']].copy()
            data.append([int(step),df.dispy.mean() ,(-1.0*df.forcey*volumes).sum()])
df = DataFrame(data,columns = ['step','dispy','forcey'])
df=df.sort_values(by=['step'])
ax = plt.subplot()
ax.plot(df.dispy.values,df.forcey.values)
ax.plot([0.0,3.0],[0.0,220.0],color='red',linestyle='dashed')
ax.set_xlabel('Average Tip Displacement [mm]')
ax.set_ylabel('Summation of Tip Forces [N]')
ax.grid(visible=True,which='both',axis='both')
ax.minorticks_on()
plt.show()
fn = datetime.now().strftime("%d%m%Y_%H%M")+'.png'
ax.figure.savefig(fn)