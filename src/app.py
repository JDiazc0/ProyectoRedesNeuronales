from flask import request, Flask
from flask_restful import Resource, Api
import keras
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
api = Api(app)

#Cargar los modelos
modelo = keras.models.load_model('modelo_prediccion')

class Neuronitas(Resource):
    """
    Función para la ruta /neuronitas
    """
    def get(self):
        """
        Método GET
        """
        return {'message': 'William Gil Clavijo 1958471 -- Samuel Ignacio Gomez 1958829 -- Jhoan Andres Diaz 1958501'}, 200

        
class predict(Resource):
    def post(self):
        file = request.files['imagen']
        if file is None:
            return {'message': 'No se ha enviado ningún archivo'}, 400
        else:
            img = Image.open(file.stream).resize((224, 224)) #Leer la imagen con PIL y redimensionar
            img = np.array(img) / 255.0 #Convertir a array y normalizar
            res = modelo.predict(img.reshape(-1,224,224,3))
            return {'result': f'El número es: {np.argmax(res[0])}',
                    'total': f'El total de predicciones es {res} '}, 200

api.add_resource(Neuronitas, '/neuronitas')
api.add_resource(predict, '/predict') # Ruta /predict

if __name__ == '__main__':
    app.run(debug=True) # Ejecutar en modo de desarrollo


        