

#archivo de preprocesamiento de los datos ionosfericos con el fin de posteriormente procesar los datos con tecnicas de ML
import pandas as pd

df = pd.read_csv('dataU.csv')

df.count()

df.info()
df['Year'] = df['Year'].map(str)
df['Day'] = df['Day'].map(str)
df['Month'] = df['Month'].map(str)

df.info()

#concatenar columnas de mes, dia, anño ya que el archivo nos los da separado y en formato DOY (365)
df['Day'] = df[['Year', 'Month','Day']].apply('-'.join, axis=1)

df['Day'] = df['Day'].astype('datetime64[ns]')

df['Hour of day'] = df['Hour of day'].map(str) #convertir tiempo de cada 20 minutos segundos a formato estandar

df['Hour of day'] = df['Hour of day'].map({
    '1.0':'01:00',
    '1.2':'01:20',
    '1.4':'01:25',
    '1.5':'01:30',
    '1.6':'01:35',
    '1.8':'01:45',
    '2.0':'02:00',
    '2.2':'02:20',
    '2.4':'02:25',
    '2.5':'02:30',
    '2.6':'02:35',
    '2.8':'02:45',
    '3.0':'3:00',
    '3.2':'3:20',
    '3.4':'3:25',
    '3.5':'3:30',
    '3.6':'3:35',
    '3.8':'3:45',
    '4.0':'04:00',
    '4.2':'04:20',
    '4.4':'4:25',
    '4.5':'4:30',
    '4.6':'04:35',
    '4.8':'04:45',
    '5.0':'5:00',
    '5.2':'5:20',
    '5.4':'5:25',
    '5.5':'5:30',
    '5.6':'5:35',
    '5.8':'5:45',
    '6.0':'06:00',
    '6.2':'06:20',
    '6.4':'06:25',
    '6.5':'06:30',
    '6.6':'06:35',
    '6.8':'06:45',
    '7.0':'07:00',
    '7.2':'07:20',
    '7.4':'07:25',
    '7.5':'07:30',
    '7.6':'07:35',
    '7.8':'07:45',
    '8.0':'08:00',
    '8.2':'08:20',
    '8.4':'08:25',
    '8.5':'08:30',
    '8.6':'08:35',
    '8.8':'08:45',
    '9.0':'09:00',
    '9.2':'09:20',
    '9.4':'09:25',
    '9.5':'09:30',
    '9.6':'09:35',
    '9.8':'09:45',
    '10.0':'10:00',
    '10.2':'10:20',
    '10.4':'10:25',
    '10.5':'10:30',
    '10.6':'10:35',
    '10.8':'10:45',
    '11.0':'11:00',
    '11.2':'11:20',
    '11.4':'11:25',
    '11.5':'11:30',
    '11.6':'11:35',
    '11.8':'11:45',
    '12.0':'12:00',
    '12.2':'12:20',
    '12.4':'12:25',
    '12.5':'12:30',
    '12.6':'12:35',
    '12.8':'12:45',
    '13.0':'13:00',
    '13.2':'13:20',
    '13.4':'13:25',
    '13.5':'13:30',
    '13.6':'13:35',
    '13.8':'13:45',
    '14.0':'14:00',
    '14.2':'14:20',
    '14.4':'14:25',
    '14.5':'14:30',
    '14.6':'14:35',
    '14.8':'14:45',
    '15.0':'15:00',
    '15.2':'15:20',
    '15.4':'15:25',
    '15.5':'15:30',
    '15.6':'15:35',
    '15.8':'15:45',
    '16.0':'16:00',
    '16.2':'16:20',
    '16.4':'16:45',
    '16.5':'16:30',
    '16.6':'16:35',
    '16.8':'16:45',
    '17.0':'17:00',
    '17.2':'17:20',
    '17.4':'17:25',
    '17.5':'17:30',
    '17.6':'17:35',
    '17.8':'17:45',
    '18.0':'18:00',
    '18.2':'18:20',
    '18.4':'18:25',
    '18.5':'18:30',
    '18.6':'18:35',
    '18.8':'18:45',
    '19.0':'19:00',
    '19.2':'19:20',
    '19.4':'19:25',
    '19.5':'19:30',
    '19.6':'19:35',
    '19.8':'19:45',
    '20.0':'20:00',
    '20.2':'20:20',
    '20.4':'20:25',
    '20.5':'20:30',
    '20.6':'20:35',
    '20.8':'20:45',
    '21.0':'21:00',
    '21.2':'21:20',
    '21.4':'21:25',
    '21.5':'21:30',
    '21.6':'21:35',
    '21.8':'21:45',
    '22.0':'22:00',
    '22.2':'22:20',
    '22.4':'22:25',
    '22.5':'22:30',
    '22.6':'22.35',
    '22.8':'22:45',
    '23.0':'23:00',
    '23.2':'23:20',
    '23.4':'23:25',
    '23.5':'23:30',
    '23.6':'23:25',
    '23.8':'23:45',
    '24.0':'00:00',
    '24.2':'00:20',
    '24.4':'24:25',
    '24.5':'00:30',
    '24.6':'00:35',
    '24.8':'00:45',
    
},na_action=None)
#borrar columnas inecesarias
df = df.drop(['Year','Day of Year','Month'], axis=1)
#rellenar datos vacios con el fin de post-procesar posteriormente
df = df.fillna(0)
#cambiar nombre de variable
data = df['magnitud']
df = df.assign(sismo=data)
#ubicar vabriable categorica para la binarización para la tecnica de Arboles de Decisión
df.loc[df['sismo'] >0,'sismo']=1
#reordenar columnas
df = df.reindex(columns=['Day','Hour of day','Latitude','Longitude','magnitud','Profundidad','Tec','Ne/cm-3','O+','N+','hmF2','foF2','VTEC','sismo'])
#guardar archivo en formato CSV
df.to_csv('train2.csv',index=False, header=True)
#Modificar parametros para realizar graficas
df['magnitud'] = df['magnitud'].map(float)
#condiciones de busqueda de datos del dataset
df2 = df[(df['Latitude']==16.84492)&(df['magnitud']>0)]


#librerias necesarias para la elaboración de graaficas
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


#guardar variables para la elaboracioón de gráficas
x = df2['hmF2']
y = df2['VTEC']
z = df2['magnitud']


#creación de gráficas en cuestion del analisis de los dias 1-8 de septiembre del 2021 sismo de Acapulco se puede ajustar a cualquier localidad cambiando la latitud
fig=plt.figure(figsize=(8,6))
axes = plt.axes(projection="3d")
axes.scatter3D(x,y,z,color="red")
axes.set_title("ANALISIS 3D TERREMOTO ACAPULCO 2021",fontsize=14,fontweight="bold")
axes.set_xlabel("HMF2")
axes.set_ylabel("TOTAL ELECTRON CONTENT")
axes.set_zlabel("Magnitud")
plt.tight_layout()
plt.show()


df3 = df[(df['Latitude']!=16.84492)&(df['magnitud']>0)]

x = df3['hmF2']
y = df3['VTEC']
z = df3['magnitud']

fig=plt.figure(figsize=(8,6))
axes = plt.axes(projection="3d")
axes.scatter3D(x,y,z,color="red")
axes.set_title("ANALISIS 3D TERREMOTO COALCOMAN 2022",fontsize=14,fontweight="bold")
axes.set_xlabel("HMF2")
axes.set_ylabel("TOTAL ELECTRON CONTENT")
axes.set_zlabel("Magnitud")
plt.tight_layout()
plt.show()



