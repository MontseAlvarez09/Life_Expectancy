from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Cargar el pipeline entrenado
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo_final.pkl')
pipeline = joblib.load(MODEL_PATH)

# Columnas que espera el pipeline (en el mismo orden del entrenamiento)
FEATURES_NUM = ['Adult Mortality', 'Income composition of resources', ' HIV/AIDS', 'Schooling', 'GDP', 'Diphtheria ']


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    errores = {}

    # --- Validación y lectura de campos ---
    def get_float(campo, nombre, minv, maxv):
        valor = request.form.get(campo, '').strip()
        if valor == '':
            errores[campo] = f'{nombre} es obligatorio.'
            return None
        try:
            v = float(valor)
        except ValueError:
            errores[campo] = f'{nombre} debe ser un número.'
            return None
        if not (minv <= v <= maxv):
            errores[campo] = f'{nombre} debe estar entre {minv} y {maxv}.'
            return None
        return v

    adult_mortality = get_float('adult_mortality', 'Mortalidad adulta', 1, 723)
    income_comp     = get_float('income_comp', 'Composición de ingresos', 0.0, 1.0)
    hiv_aids        = get_float('hiv_aids', 'Muertes por VIH/SIDA', 0.1, 50.6)
    schooling       = get_float('schooling', 'Escolaridad', 0.0, 20.7)
    gdp             = get_float('gdp', 'PIB per cápita', 1.0, 120000.0)
    diphtheria      = get_float('diphtheria', 'Cobertura Difteria', 2.0, 99.0)

    # Status se sigue leyendo y validando para mostrarlo en pantalla,
    # pero no se le pasa al modelo porque el pipeline no fue entrenado con esta columna
    status = request.form.get('status', '').strip()
    if status not in ('Developed', 'Developing'):
        errores['status'] = 'Selecciona un nivel de desarrollo válido.'

    if errores:
        return render_template('index.html', errores=errores, form=request.form)

    # --- Construcción del DataFrame de entrada (solo las 6 columnas del pipeline) ---
    entrada = pd.DataFrame([{
        'Adult Mortality':                    adult_mortality,
        'Income composition of resources':    income_comp,
        ' HIV/AIDS':                          hiv_aids,
        'Schooling':                          schooling,
        'GDP':                                gdp,
        'Diphtheria ':                        diphtheria
    }])

    # --- Predicción ---
    prediccion = pipeline.predict(entrada)[0]
    prediccion = round(float(prediccion), 2)

    return render_template('index.html', prediccion=prediccion, form=request.form)


if __name__ == '__main__':
    app.run(debug=False)
