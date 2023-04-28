#librerias a usar
import numpy as np
from flask import Flask, request, jsonify, render_template,redirect,Response
import pickle
import csv
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from flask_bootstrap import Bootstrap4

# Create flask app
flask_app = Flask(__name__)
bootstrap = Bootstrap4(flask_app)
model = pickle.load(open("entrenamiento.pickle", "rb"))#carga del entrenamiento de arboles de decision
regression = pickle.load(open("regression.pickle", "rb"))#carga del entrenamiento arboles de regresion


@flask_app.route("/")
def Home():
    return render_template("index.html")
###algoritmo de arboles de decision
@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "La probabibilidad de sismo es de {}".format(prediction) + "con un porcentaje de 87% de precision")

#conversion de CSV  a BD con postgreSQL 
@flask_app.route('/cargar_csv', methods=['GET', 'POST'])
def cargar_csv():
    if request.method == 'POST':
        # obtener el archivo CSV cargado en el formulario
        archivo = request.files['archivo']
        # leer el contenido del archivo CSV
        contenido = archivo.read().decode('utf-8').splitlines()
        # conectar a la base de datos
        conexion = psycopg2.connect(
            dbname= "ionosferic",
            user="postgres",
            password="buffon091294",
            host="localhost",
            port=5432
        )
        cursor = conexion.cursor()
        # iterar sobre las filas del archivo CSV y guardarlas en la base de datos
        for fila in csv.reader(contenido):
            cursor.execute("INSERT INTO datos3 (dia, hora, latitude, longitude, magnitud, profundidad,tec, plasma, o, n, hmf2, fof2, vtec) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", fila)
        # confirmar los cambios en la base de datos
        conexion.commit()
        # cerrar la conexión a la base de datos
        cursor.close()
        conexion.close()
        # redirigir a la página de inicio
        return redirect('/')
    else:
        # mostrar el formulario para cargar el archivo CSV
        return render_template('cargar_csv.html')

@flask_app.route('/grafica-3d')
def grafica_3d():
    # leer los datos del archivo CSV utilizando pandas
    datos = pd.read_csv('train.csv')
    datos['magnitud'] = datos['magnitud'].map(float)
    datos = datos[(datos['Latitude']==16.84492)&(datos['magnitud']>0)]
    
    # crear la figura y los ejes de la gráfica en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # agregar los datos a la gráfica en 3D
    ax.scatter(datos['hmF2'], datos['VTEC'], datos['magnitud'])
    # personalizar la gráfica en 3D
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')
    # guardar la imagen de la gráfica en 3D en un archivo PNG
    plt.savefig('grafica_3d.png')
    # mostrar la imagen de la gráfica en 3D en la plantilla HTML
    return render_template('grafica_3d.html')


if __name__ == "__main__":
    flask_app.run(debug=True)
##########prediccion con algoritmo de arboles de regresion
@flask_app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    if request.method == 'POST':
        # obtener los valores del formulario HTML
        x1 = float(request.form['x1'])
        x2 = float(request.form['x2'])
        x3 = float(request.form['x3'])

        # generar la predicción utilizando el modelo
        prediccion = regression.predict([[x1, x2, x3]])

        return render_template('formulario.html', prediccion="La posible magnitud del movimiento sismico en base a los parametros es: {}".format(prediccion))
    else:
        return render_template('formulario.html')
#conexion a la BD
conexion = psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='ionosferic',
        user='postgres',
        password=''
    )

@flask_app.route('/resultados')
def mostrar_resultados():
    cur = conexion.cursor()
    cur.execute('SELECT * FROM datos3')
    resultados = cur.fetchall()
    return render_template('resultados.html', resultados=resultados) 



    