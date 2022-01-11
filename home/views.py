from django.shortcuts import render
from django.http import JsonResponse

from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import TaskSerializer

from .models import Task

@api_view(['GET'])
def index(request):
  api_urls = {
    'List':'/task-list/',
    'Detail View':'/task-detail/<str:pk>/',
    'Create':'/task-create/',
    'Update':'/task-update/<str:pk>/',
    'Delete':'/task-delete/<str:pk>/',
  }

  return Response(api_urls)

@api_view(['GET'])
def taskList(request):
  tasks = Task.objects.all()
  serializer = TaskSerializer(tasks, many=True)
  return Response(serializer.data)

@api_view(['GET'])
def taskDetail(request, pk):
  tasks = Task.objects.get(id=pk)
  serializer = TaskSerializer(tasks, many=False)
  return Response(serializer.data)

@api_view(['POST'])
def taskCreate(request):
  print("1:")
  print(request.data['title'])
  serializer = TaskSerializer(data=request.data)
  print("2:")
  print(serializer)

  if serializer.is_valid():
    serializer.save()

  return Response(serializer.data)

@api_view(['POST'])
def taskUpdate(request, pk):
  task = Task.objects.get(id=pk)
  serializer = TaskSerializer(instance=task, data=request.data)

  if serializer.is_valid():
    serializer.save()

  return Response(serializer.data)

@api_view(['DELETE'])
def taskDelete(request, pk):
  task = Task.objects.get(id=pk)
  task.delete()

  return Response('Item succsesfully delete!')


@api_view(['GET', 'POST'])
def load_data(request):

  """ Código Pandas """
  import pathlib
  import matplotlib.pyplot as plt
  import pandas as pd
  import seaborn as sns
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers
  import numpy as np
  import json

  workbook_name = request.FILES['dir_excel']

  if(".xlsx" in str(workbook_name)):
    dataframe = pd.read_excel(workbook_name, engine='openpyxl', sheet_name=0)

  if(".csv" in str(workbook_name)):
    dataframe = pd.read_csv(workbook_name)

  if(".txt" in str(workbook_name)):
    dataframe = pd.read_csv(workbook_name, sep=";")


  # Identifica las columnas datatime con el fin de eliminar las columnas de fechas
  for col in dataframe.columns:
    if dataframe[col].dtype == 'object':
        try:
            # Si la columna tiene el formato datetime, reemplaza la columna con type oject a datetime64
            dataframe[col] = pd.to_datetime(dataframe[col])
        except:
            pass
  #print('\nVisualización del DataSet por tipo:')
  #print(dataframe.dtypes)


  # Se identifican las columnas según su tipo
  columns_datetime = dataframe.dtypes[dataframe.dtypes == 'datetime64[ns]'].index.to_list()

  # Elimina las columnas datetime para tener un correcto select del target
  dataframe_sin_datetime = dataframe.drop(columns=columns_datetime)


  """ Convierte el Dataframe sin fechas en json para extraer el name de sus columnas"""
  dataframe_sin_datetime = json.loads(dataframe_sin_datetime.to_json(orient='records'))


  # Se carga otra vez el df (se producía un error en la visualización de las fechas ej. 10-12-2021 -> 10122021)
  if(".xlsx" in str(workbook_name)):
    dataframe = pd.read_excel(workbook_name, engine='openpyxl', sheet_name=0)

  if(".csv" in str(workbook_name)):
    for column in columns_datetime:
      dataframe[column] = dataframe[column].astype(str)

  if(".txt" in str(workbook_name)):
    for column in columns_datetime:
      dataframe[column] = dataframe[column].astype(str)


  # Convierte las columnas de type bool a str o int, según su contenido
  for name_column, value_column in dataframe.items():
    if(dataframe[name_column].dtypes == bool):
      if(isinstance(dataframe[name_column].dtypes, object) == True):
        dataframe[name_column] = dataframe[name_column].astype(str)
      else:
        dataframe[name_column] = dataframe[name_column].astype(np.float64)


  # Elimina todas las filas con datos NaN (se evitan errores de compilación)
  dataframe = dataframe.dropna()


  """ Crea la columna index """
  dataframe[''] = range(1, len(dataframe) + 1)
  """ Se ubica la última columna en la primera posición  """
  cols = list(dataframe)
  cols = [cols[-1]] + cols[:-1]
  dataframe = dataframe[cols]


  """ Convierte el Dataframe en json """
  datos_json = json.loads(dataframe.to_json(orient='records'))


  data = [{
    'datos_json': datos_json,
    'dataframe_sin_datetime': dataframe_sin_datetime
  }]

  return Response(data)

@api_view(['GET', 'POST'])
def detect_type_neuronal(request):
  """ Carga de librerías """
  import pandas as pd
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers
  import numpy as np
  import json
  from tensorflow.keras import layers
  from tensorflow.keras.layers.experimental import preprocessing

  # Convierte los datos json en un dataframe
  dataframe = pd.json_normalize(request.data["datos"])

  # Elimina la columna index creada en la func anterior para visualización en html
  dataframe.drop([''], axis=1, inplace=True)

  if(len(dataframe[request.data["target_predecir"]].unique()) == 2):

    """ Se envían los datos al Frontend """
    data = [{
        'tipo_red_neuronal': 'binary', # Tipo de Red Neuronal
        'hidden_cap1': 32
    }]

  else:
    if(dataframe[request.data["target_predecir"]].dtype != object):

      """ Se envían los datos al Frontend """
      data = [{
        'tipo_red_neuronal': 'number', # Tipo de Red Neuronal
        'hidden_cap1': 32
      }]

    if(dataframe[request.data["target_predecir"]].dtype == object):

      """ Se envían los datos al Frontend """
      data = [{
          'tipo_red_neuronal': 'string', # Tipo de Red Neuronal
          'hidden_cap1': 256
      }]

  return Response(data)

@api_view(['GET', 'POST'])
def run_red_neuronal(request):

  """ Carga de librerías """
  import pathlib
  import matplotlib # Necesario para crear gráficos (ej. gráfico de entrenamiento neuronal)
  matplotlib.use('Agg') # Necesario para exportar el gráfico matplotlib de entrenamiento en svg y enviarlo al frontend
  import matplotlib.pyplot as plt
  import pandas as pd
  import seaborn as sns
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers
  import numpy as np
  import json

  #from sklearn.model_selection import train_test_split
  from tensorflow.keras import layers
  from tensorflow.keras.layers.experimental import preprocessing

  # Convierte los datos json en un dataframe
  dataframe = pd.json_normalize(request.data["datos"])

  # Elimina la columna index creada en la func anterior para visualización en html
  dataframe.drop([''], axis=1, inplace=True)

  # Se debe eliminar el archivo previamente porque react al reemplazar algún archivo inmediatamente refresca toda la página web y se perdían los datos.
  import os
  # Si el archivo existe, lo elimina (para evitar errores)
  if os.path.isfile('./frontend/public/files/Connectivity_graph.png'):
    os.remove('./frontend/public/files/Connectivity_graph.png')


  # Si la carpeta existe, lo elimina (para evitar errores)
  if os.path.isfile('./frontend/public/files/modelo_red_neuronal.zip'):
    os.remove('./frontend/public/files/modelo_red_neuronal.zip')


  if(len(dataframe[request.data["target_predecir"]].unique()) == 2):
    # Se Reemplazan los espacios en el nombre de las columna por guiones bajos
    dataframe.columns = dataframe.columns.str.replace(' ','_')

    """ Convierte la columna TARGET en una columna numérica """ 
    # necesario para la transformación
    dataframe[request.data["target_predecir"]] = pd.Categorical(dataframe[request.data["target_predecir"]])
    # Se agregan los códigos al dataframe
    dataframe['code_target'] = dataframe[request.data["target_predecir"]].cat.codes


    """ Metodos que se utilizan para identificar el target por números """
    # Se crea un dataframe con los códigos del target
    class_names = dataframe[['code_target', request.data["target_predecir"]]]
    class_names = class_names.sort_values('code_target')
    class_names_unique = class_names[['code_target', request.data["target_predecir"]]].drop_duplicates()

    # Almacena los valores del target en una lista
    class_names = class_names_unique[request.data["target_predecir"]].values.tolist()


    """ Número de salidas de la Red Neural (cantidad de opciones proporcionadas por el target) """
    cantidad_posibles_predicciones = len(class_names) - 1


    """ Se identifican las columnas datatime del dataframe """
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            try:
                dataframe[col] = pd.to_datetime(dataframe[col])
            except ValueError:
                pass
    #print('\n', dataframe.dtypes)


    # Se almacenan las columnas en variables según tipo de datos
    columns_numeric = dataframe.dtypes[(dataframe.dtypes == 'int64') | (dataframe.dtypes == 'float64') | (dataframe.dtypes == 'int8')].index.to_list()
    columns_string = dataframe.dtypes[dataframe.dtypes == 'object'].index.to_list()
    columns_datetime = dataframe.dtypes[dataframe.dtypes == 'datetime64[ns]'].index.to_list()

    # Elimina el target de las columnas numéricas
    to_remove = 'code_target'
    # Si se encuentra la columna en la lista, se elimina
    if to_remove in columns_numeric:
        columns_numeric.remove(to_remove)

    # Elimina el target de las columnas categóricas
    to_remove = 'code_target'
    # Si se encuentra la columna en la lista, se elimina
    if to_remove in columns_string:
        columns_string.remove(to_remove)

    # Se unen las columnas numéricas y categóricas
    all_columns_usable = columns_numeric + columns_string # Se utilizarán para crear el formulario de consultas
    #print("\nColumnas numéricas: ", columns_numeric)
    #print("\nColumnas características:", columns_string)
    #print("\nColumnas datetime: ", columns_datetime)
    #print("\nTodas las Columnas sin el target: ", all_columns_usable)

    # Elimina las columnas DateTime del dataframe
    for column in columns_datetime:
        dataframe.drop([column], axis=1, inplace=True)


    """ Obtienes todos los valores de las columnas categóricas para utilizar en el form y poder mostrar las opciones en el select """
    columnas_categoricas_onehot_json = {}
    for column in columns_string:
        columnas_categoricas_onehot_json[column]= dataframe[column].sort_values().unique()


    """ Preparación para el entrenamiento """
    # División del dataset en datos de entrenamiento y prueba
    train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
    len_train_dataset = len(train) # Se almacena en una variable la cant. de datos de entrenamiento para enviarlo al front
    len_val_dataset = len(val) # Se almacena en una variable la cant. de datos de prueba para enviarlo al front
    len_test_dataset = len(test) # Se almacena en una variable la cant. de datos de prueba para enviarlo al front
    #print(len(train), '\n training examples')
    #print(len(val), '\n validation examples')
    #print(len(test), '\n test examples')

    # Crea una canalización de entrada usando tf.data
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
      dataframe = dataframe.copy()
      labels = dataframe.pop('code_target')
      ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
      if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
      ds = ds.batch(batch_size)
      ds = ds.prefetch(batch_size)
      return ds

    # Demuestre el uso de capas de preprocesamiento
    def get_normalization_layer(name, dataset):
      # Create a Normalization layer for our feature.
      normalizer = preprocessing.Normalization()
      # Prepare a Dataset that only yields our feature.
      feature_ds = dataset.map(lambda x, y: x[name])
      # Learn the statistics of the data.
      normalizer.adapt(feature_ds)
      return normalizer

    # Columnas categóricas
    def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
      # Create a StringLookup layer which will turn strings into integer indices
      if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
      else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)
      # Prepare a Dataset that only yields our feature
      feature_ds = dataset.map(lambda x, y: x[name])
      # Learn the set of possible values and assign them a fixed integer index.
      index.adapt(feature_ds)
      # Create a Discretization for our integer indices.
      encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())
      # Apply one-hot encoding to our indices. The lambda function captures the
      # layer so we can use them, or include them in the functional model later.
      return lambda feature: encoder(index(feature))

    # Canalización de entrada
    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


    """ NUMERIC FEATURES """
    all_inputs = []
    encoded_features = []

    for header in columns_numeric:
      numeric_col = tf.keras.Input(shape=(1,), name=header)
      normalization_layer = get_normalization_layer(header, train_ds)
      encoded_numeric_col = normalization_layer(numeric_col)
      all_inputs.append(numeric_col)
      encoded_features.append(encoded_numeric_col)


    """ CATEGORICAL FEATURES """
    for header in columns_string:
      categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
      encoding_layer = get_category_encoding_layer(name=header,
                                                  dataset=train_ds,
                                                  dtype='string',
                                                  max_tokens=5)
      encoded_categorical_col = encoding_layer(categorical_col)
      all_inputs.append(categorical_col)
      encoded_features.append(encoded_categorical_col)


    """ MODELO DE LA RED NEURONAL """
    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(cantidad_posibles_predicciones)(x)

    model = tf.keras.Model(all_inputs, output)


    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])


    """ Gráfico de conectividad """
    tf.keras.utils.plot_model(model, to_file='./frontend/public/files/Connectivity_graph.png', show_shapes=True, rankdir="LR")


    """ ENTRENAMIENTO DEL MODELO """
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    #history = model.fit(train_ds, epochs=200, validation_data=val_ds, verbose=1, callbacks=[early_stop]) # Local
    history = model.fit(train_ds, epochs=100, validation_data=val_ds, verbose=1, callbacks=[early_stop]) # Producción


    ### Gráficos del entrenamiento
    from io import StringIO

    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    #plt.show()

    # Código para exportar en SVG
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    graph1 = imgdata.getvalue()

    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    #plt.show()

    # Código para exportar en SVG
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    graph2 = imgdata.getvalue()


    # Visualización del progreso de entrenamiento del modelo usando las estadísticas almacenadas en el objeto history.
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    dataframe_hist_tail = hist.tail() # Almacena los resultados en una variable para enviarlos al frontend
    hist.tail()

    # Transforma el Dataframe con los resultados del entrenamiento a json para enviarlos al frontend
    datos_hist_tail = json.loads(dataframe_hist_tail.to_json(orient='records'))

    # Resultados del entrenamiento con datos de testeo
    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)
    accuracy = accuracy * 100


    """ Sección de Prueba """
    # Quita la columna a predecir (target) del dataset de testeo y finalmente almacena los datos target en una lista
    test_labels = test.pop('code_target')

    # Se realiza la predicción en todos los datos de pruebas
    predicciones = model.predict(test_ds)
    predicciones = tf.nn.sigmoid(predicciones)

    predicciones_number = []
    for i, logits in enumerate(predicciones):
        class_idx = logits[0].numpy().round().astype(int)
        predicciones_number.append(class_idx)


    # Se almacenan las predicciones en un Dataframe
    predicciones_df = pd.DataFrame(predicciones_number, columns = ['Predicciones'])

    # Se Almacena los resultados originales en un Dataframe
    test_labels_df = pd.DataFrame(test_labels.to_list(), columns = ['Original'])

    # Unión de los Dataframe creados anteriormente
    df_originals_predictions = test_labels_df.join(predicciones_df)
    

    # Convierte el código asignado al texto original 
    for index, row in df_originals_predictions.iterrows():
      df_originals_predictions['Original'][index] = class_names[row['Original']]
      df_originals_predictions['Predicciones'][index] = class_names[row['Predicciones']]


    """ Matrix de confusión """
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(df_originals_predictions['Original'].to_list(),df_originals_predictions['Predicciones'].to_list(), labels=class_names)
    # Transforma la matrix de confusión en un dataframe
    df_matrix = pd.DataFrame(matrix, columns=class_names, index=class_names)

    fig = plt.figure()
    matrix_graph = sns.heatmap(df_matrix, annot=True, cbar=False, cmap='Blues')
    matrix_graph.set(xlabel='Predicted',ylabel='Original')
    matrix_graph.set_yticklabels(class_names, rotation=0, va="center")

    # Código para exportar en SVG
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    graph3 = imgdata.getvalue()


    # Se crea el index del Dataframe anterior para una mejor visualización en el frontend
    # Crea la columna index
    df_originals_predictions[''] = range(1, len(df_originals_predictions) + 1)
    # Se ubica la última columna en la primera posición
    cols = list(df_originals_predictions)
    cols = [cols[-1]] + cols[:-1]
    df_originals_predictions = df_originals_predictions[cols]

    # Transforma el Dataframe a json para enviarlos al frontend
    df_originals_predictions_json = json.loads(df_originals_predictions.to_json(orient='records'))


    """ Guardando la Red Neuronal """
    model.save('modelo_red_neuronal')

    # Transforma la carpeta del modelo neuronal en un zip para poder descargarla por html
    import shutil
    shutil.make_archive('modelo_red_neuronal', 'zip', 'modelo_red_neuronal')
    
    # Mueve el archivo.zip a las carpetas de react, fue la única forma de evitar que la página se actualizara
    from pathlib import Path
    Path("modelo_red_neuronal.zip").rename("./frontend/public/files/modelo_red_neuronal.zip")


    """ Se envían los datos al Frontend """
    data = [{
        'target_predecir_after_train': request.data["target_predecir"], # Target: valor a predecir por la Red Neuronal
        'columns_numeric': columns_numeric, # Todas las variables utilizadas (sin fechas y sin el target)
        'columnas_categoricas_onehot_json': columnas_categoricas_onehot_json, # Columnas categóricas con sus respectivas colunas unique()
        'len_train_dataset': len_train_dataset, # Se utiliza para hacer la pila con la partición de los datos
        'len_val_dataset': len_val_dataset, # Se utiliza para hacer la pila con la partición de los datos
        'len_test_dataset': len_test_dataset, # Se utiliza para hacer la pila con la partición de los datos
        'grafico1': graph1, # Gráfico 1 del entrenamiento
        'grafico2': graph2, # Gráfico 2 del entrenamiento
        'grafico3': graph3, # matrix de confusión
        #'graph_conect': graph_conect, # Gráfico de conectividad (diseño de la Red Neuronal)
        'datos_hist_tail': datos_hist_tail, # Datos del entrenamiento para ser visualizados en una tabla
        'loss': loss, # Datos del entrenamiento
        'mae': accuracy, # Datos del entrenamiento
        'df_originals_predictions_json': df_originals_predictions_json, # Datos del target original vs predicciones, en el frontend se visualizan en una tabla
        'tipo_red_neuronal': 'binary', # Tipo de Red Neuronal
        'class_names': class_names, # almacena lo que significa para código en una predicción string
    }]

  else:
    if((dataframe[request.data["target_predecir"]].dtype != object) and (dataframe[request.data["target_predecir"]].dtype == 'float64')):
      # Se Reemplazan los espacios en el nombre de las columna por guiones bajos
      dataframe.columns = dataframe.columns.str.replace(' ','_')

      #print(isinstance(dataframe[request.data["target_predecir"]], int))

      # Se identifican las columnas datatime del dataframe
      for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            try:
                dataframe[col] = pd.to_datetime(dataframe[col])
            except ValueError:
                pass
      #print('\nVisualización por tipo de datos de cada columna del dataframe:')
      print(dataframe.dtypes)

      #### Se almacenan las columnas en variables según tipo de datos
      columns_numeric = dataframe.dtypes[(dataframe.dtypes == 'int64') | (dataframe.dtypes == 'float64')].index.to_list()
      columns_string = dataframe.dtypes[dataframe.dtypes == 'object'].index.to_list()
      columns_datetime = dataframe.dtypes[dataframe.dtypes == 'datetime64[ns]'].index.to_list()

      all_columns_usable = columns_numeric + columns_string # Se utilizarán para crear el formulario de consultas
      all_columns_usable.remove(request.data["target_predecir"]) # Elimina el target de la lista

      #print("Columnas numéricas: ", columns_numeric)
      #print("Columnas características:", columns_string)
      #print("Columnas datetime: ", columns_datetime)
      #print("Todas las Columnas sin el target: ", all_columns_usable)

      #### Elimina las columnas DateTime del dataframe
      for column in columns_datetime:
        dataframe.drop([column], axis=1, inplace=True)

      #### Obtienes todos los valores de las columnas categóricas para utilizar en el form y poder mostrar las opciones en el select
      columnas_categoricas_onehot_json = {}
      for column in columns_string:
          columnas_categoricas_onehot_json[column]= dataframe[column].sort_values().unique()

      #### División del dataset en datos de entrenamiento y prueba
      train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
      len_train_dataset = len(train) # Se almacena en una variable la cant. de datos de entrenamiento para enviarlo al front
      len_val_dataset = len(val) # Se almacena en una variable la cant. de datos de prueba para enviarlo al front
      len_test_dataset = len(test) # Se almacena en una variable la cant. de datos de prueba para enviarlo al front

      #print(len(train), 'training examples')
      #print(len(val), 'validation examples')
      #print(len(test), 'test examples')

      #### Crea una canalización de entrada usando tf.data
      def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop(request.data["target_predecir"])
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
          ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

      #### Demuestre el uso de capas de preprocesamiento
      def get_normalization_layer(name, dataset):
        # Create a Normalization layer for our feature.
        normalizer = preprocessing.Normalization()
        # Prepare a Dataset that only yields our feature.
        feature_ds = dataset.map(lambda x, y: x[name])
        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)
        return normalizer


      #### Columnas categóricas
      def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
        # Create a StringLookup layer which will turn strings into integer indices
        if dtype == 'string':
          index = preprocessing.StringLookup(max_tokens=max_tokens)
        else:
          index = preprocessing.IntegerLookup(max_values=max_tokens)

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)

        # Create a Discretization for our integer indices.
        encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

        # Apply one-hot encoding to our indices. The lambda function captures the
        # layer so we can use them, or include them in the functional model later.
        return lambda feature: encoder(index(feature))


      ### canalización de entrada
      batch_size = 256
      train_ds = df_to_dataset(train, batch_size=batch_size)
      val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
      test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

      #### Numeric features.
      columns_numeric.remove(request.data["target_predecir"])

      all_inputs = []
      encoded_features = []

      # Numerical features.
      for header in columns_numeric:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)


      #### Categorical features encoded as string.
      for header in columns_string:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(name=header,
                                                    dataset=train_ds,
                                                    dtype='string',
                                                    max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)


      """ MODELO DE LA RED NEURONAL """
      all_features = tf.keras.layers.concatenate(encoded_features)
      #x = tf.keras.layers.Dense(64, activation="relu")(all_features) # Local
      x = tf.keras.layers.Dense(32, activation="relu")(all_features)  # Producción
      #x = tf.keras.layers.Dense(64, activation="relu")(x) # Local
      #x = tf.keras.layers.Dropout(0.5)(x) # Local
      output = tf.keras.layers.Dense(1)(x)

      model = tf.keras.Model(all_inputs, output)


      #optimizer = tf.keras.optimizers.RMSprop(0.0001) # Local
      optimizer = tf.keras.optimizers.RMSprop(0.001) # Producción

      model.compile(optimizer=optimizer,
                    loss='mse',
                    metrics=['mae', 'mse'])


      """ Gráfico de conectividad """
      tf.keras.utils.plot_model(model, to_file='./frontend/public/files/Connectivity_graph.png', show_shapes=True, rankdir="LR")


      """ ESTRUCTURACIÓN PREVIA AL ENTRENAMIENTO DEL MODELO """
      #### Función que generará el gráfico de entrenamiento
      def plot_history(history):
          from io import StringIO
          hist = pd.DataFrame(history.history)
          hist['epoch'] = history.epoch

          #plt.figure()
          fig = plt.figure()

          plt.xlabel('Epoch')
          plt.ylabel('Mean Abs Error')
          plt.plot(hist['epoch'], hist['mae'],
                  label='Train Error')
          plt.plot(hist['epoch'], hist['val_mae'],
                  label = 'Val Error')
          #plt.ylim([0,5])
          plt.title("")
          plt.legend()

          # SVG
          imgdata = StringIO()
          fig.savefig(imgdata, format='svg')
          imgdata.seek(0)
          graph1 = imgdata.getvalue()

          #plt.figure()
          fig = plt.figure()
          plt.xlabel('Epoch')
          plt.ylabel('Mean Square Error')
          plt.plot(hist['epoch'], hist['mse'],
                  label='Train Error')
          plt.plot(hist['epoch'], hist['val_mse'],
                  label = 'Val Error')
          #plt.ylim([0,20])
          plt.title("")
          plt.legend()

          # SVG
          imgdata = StringIO()
          fig.savefig(imgdata, format='svg')
          imgdata.seek(0)
          graph2 = imgdata.getvalue()

          #plt.show()
          return [graph1, graph2]


      """ ENTRENAMIENTO DEL MODELO """
      early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=90)

      #history = model.fit(train_ds, epochs=1000, # local
      history = model.fit(train_ds, epochs=350, # Producción
                          validation_data=val_ds, verbose=0, callbacks=[early_stop])


      #### Visualizar los gráficos del entrenamiento
      plots_history = plot_history(history)


      #### Visualización del progreso de entrenamiento del modelo usando las estadísticas almacenadas en el objeto history.
      hist = pd.DataFrame(history.history)
      hist['epoch'] = history.epoch
      dataframe_hist_tail = hist.tail() # Almacena los resultados en una variable para enviarlos al frontend
      hist.tail()


      #### Transforma el Dataframe con los resultados del entrenamiento a json para enviarlos al frontend
      datos_hist_tail = json.loads(dataframe_hist_tail.to_json(orient='records'))


      #### Resultados del entrenamiento con datos de testeo
      loss, mae, mse = model.evaluate(test_ds)
      #print("loss", loss)
      #print("mae", mae)
      #print("mse", mse)

      """ FIN DEL ENTRENAMIENTO """


      """ SECCIÓN DE PRUEBAS """
      #### Quita la columna a predecir (target) del dataset de testeo y finalmente almacena los datos target en una lista  
      test_labels = test.pop(request.data["target_predecir"])

      #### Se realiza la predicción en todos los datos de pruebas
      predicciones = model.predict(test_ds).flatten()

      #### Se almacenan las predicciones en un Dataframe
      #### convierte las columnas en float
      predicciones_df = pd.DataFrame(predicciones, columns = ['Predicciones'])
      predicciones_df = predicciones_df.round(2)

      #### Se Almacena los resultados originales en un Dataframe
      test_labels_df = pd.DataFrame(test_labels.to_list(), columns = ['Original'])

      #### Unión de los Dataframe creados anteriormente
      df_originals_predictions = test_labels_df.join(predicciones_df)


      #### Se crea el index del Dataframe anterior para una mejor visualización en el frontend
      # Crea la columna index
      df_originals_predictions[''] = range(1, len(df_originals_predictions) + 1)
      # Se ubica la última columna en la primera posición
      cols = list(df_originals_predictions)
      cols = [cols[-1]] + cols[:-1]
      df_originals_predictions = df_originals_predictions[cols]

      # Transforma el Dataframe a json para enviarlos al frontend
      df_originals_predictions_json = json.loads(df_originals_predictions.to_json(orient='records'))


      """ Guardando la Red Neuronal """
      model.save('modelo_red_neuronal')

      # Transforma la carpeta del modelo neuronal en un zip para poder descargarla por html
      import shutil
      shutil.make_archive('modelo_red_neuronal', 'zip', 'modelo_red_neuronal')
      
      # Mueve el archivo.zip a las carpetas de react, fue la única forma de evitar que la página se actualizara
      from pathlib import Path
      Path("modelo_red_neuronal.zip").rename("./frontend/public/files/modelo_red_neuronal.zip")


      ## Se envían los datos al frontend
      data = [{
        'grafico1': plots_history[0], # Gráfico 1 del entrenamiento
        'grafico2': plots_history[1], # Gráfico 2 del entrenamiento
        #'graph_conect': graph_conect, # Gráfico de conectividad (diseño de la Red Neuronal)
        'datos_hist_tail': datos_hist_tail, # Datos del entrenamiento para ser visualizados en una tabla
        'loss': loss, # Datos del entrenamiento
        'mae': mae, # Datos del entrenamiento
        'mse': mse, # Datos del entrenamiento
        'df_originals_predictions_json': df_originals_predictions_json, # Datos del target original vs predicciones, en el frontend se visualizan en una tabla
        'columns_numeric': columns_numeric, # Todas las variables utilizadas (sin fechas y sin el target)
        'columnas_categoricas_onehot_json': columnas_categoricas_onehot_json, # Columnas categóricas con sus respectivas colunas unique()
        'target_predecir_after_train': request.data["target_predecir"], # Target: valor predicho por la Red Neuronal
        'len_train_dataset': len_train_dataset, # Se utiliza para hacer la pila con la partición de los datos
        'len_val_dataset': len_val_dataset, # Se utiliza para hacer la pila con la partición de los datos
        'len_test_dataset': len_test_dataset, # Se utiliza para hacer la pila con la partición de los datos
        'tipo_red_neuronal': 'number', # Tipo de Red Neuronal
      }]

    else:
      # Se Reemplazan los espacios en el nombre de las columna por guiones bajos
      dataframe.columns = dataframe.columns.str.replace(' ','_')


      """ Convierte la columna TARGET en una columna numérica """ 
      # necesario para la transformación
      dataframe[request.data["target_predecir"]] = pd.Categorical(dataframe[request.data["target_predecir"]])
      # Se agregan los códigos al dataframe
      dataframe['code_target'] = dataframe[request.data["target_predecir"]].cat.codes


      """ Metodos que se utilizan para identificar el target por números """
      # Se crea un dataframe con los códigos del target
      class_names = dataframe[['code_target', request.data["target_predecir"]]]
      class_names = class_names.sort_values('code_target')
      class_names_unique = class_names[['code_target', request.data["target_predecir"]]].drop_duplicates()
      #print(class_names_unique)
      # Almacena los valores del target en una lista
      class_names = class_names_unique[request.data["target_predecir"]].values.tolist()
      #print(class_names)

      """ Número de salidas de la Red Neural (cantidad de opciones proporcionadas por el target) """
      cantidad_posibles_predicciones = len(class_names)


      """ Se identifican las columnas datatime del dataframe """
      for col in dataframe.columns:
          if dataframe[col].dtype == 'object':
              try:
                  dataframe[col] = pd.to_datetime(dataframe[col])
              except ValueError:
                  pass
      #print('\n', dataframe.dtypes)

      # Se almacenan las columnas en variables según tipo de datos
      columns_numeric = dataframe.dtypes[(dataframe.dtypes == 'int64') | (dataframe.dtypes == 'float64') | (dataframe.dtypes == 'int8')].index.to_list()
      columns_string = dataframe.dtypes[dataframe.dtypes == 'object'].index.to_list()
      columns_datetime = dataframe.dtypes[dataframe.dtypes == 'datetime64[ns]'].index.to_list()

      # Elimina el target de las columnas numéricas
      to_remove = 'code_target'
      # Si se encuentra la columna en la lista, se elimina
      if to_remove in columns_numeric:
          columns_numeric.remove(to_remove)

      # Elimina el target de las columnas categóricas
      to_remove = 'code_target'
      # Si se encuentra la columna en la lista, se elimina
      if to_remove in columns_string:
          columns_string.remove(to_remove)

      # Se unen las columnas numéricas y categóricas
      all_columns_usable = columns_numeric + columns_string # Se utilizarán para crear el formulario de consultas
      #print("\nColumnas numéricas: ", columns_numeric)
      #print("\nColumnas características:", columns_string)
      #print("\nColumnas datetime: ", columns_datetime)
      #print("\nTodas las Columnas sin el target: ", all_columns_usable)

      # Elimina las columnas DateTime del dataframe
      for column in columns_datetime:
          dataframe.drop([column], axis=1, inplace=True)


      """ Obtienes todos los valores de las columnas categóricas para utilizar en el form y poder mostrar las opciones en el select """
      columnas_categoricas_onehot_json = {}
      for column in columns_string:
          columnas_categoricas_onehot_json[column]= dataframe[column].sort_values().unique()

      """ Preparación para el entrenamiento """
      # División del dataset en datos de entrenamiento y prueba
      train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
      len_train_dataset = len(train) # Se almacena en una variable la cant. de datos de entrenamiento para enviarlo al front
      len_val_dataset = len(val) # Se almacena en una variable la cant. de datos de prueba para enviarlo al front
      len_test_dataset = len(test) # Se almacena en una variable la cant. de datos de prueba para enviarlo al front
      #print(len(train), '\n training examples')
      #print(len(val), '\n validation examples')
      #print(len(test), '\n test examples')

      # Crea una canalización de entrada usando tf.data
      def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('code_target')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
          ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

      # Demuestre el uso de capas de preprocesamiento
      def get_normalization_layer(name, dataset):
        # Create a Normalization layer for our feature.
        normalizer = preprocessing.Normalization()
        # Prepare a Dataset that only yields our feature.
        feature_ds = dataset.map(lambda x, y: x[name])
        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)
        return normalizer

      # Columnas categóricas
      def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
        # Create a StringLookup layer which will turn strings into integer indices
        if dtype == 'string':
          index = preprocessing.StringLookup(max_tokens=max_tokens)
        else:
          index = preprocessing.IntegerLookup(max_values=max_tokens)
        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])
        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)
        # Create a Discretization for our integer indices.
        encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())
        # Apply one-hot encoding to our indices. The lambda function captures the
        # layer so we can use them, or include them in the functional model later.
        return lambda feature: encoder(index(feature))

      # Canalización de entrada
      batch_size = 32
      train_ds = df_to_dataset(train, batch_size=batch_size)
      val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
      test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


      """ NUMERIC FEATURES """
      all_inputs = []
      encoded_features = []

      # Numerical features.
      for header in columns_numeric:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)


      """ CATEGORICAL FEATURES """
      for header in columns_string:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(name=header,
                                                    dataset=train_ds,
                                                    dtype='string',
                                                    max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)


      """ MODELO DE LA RED NEURONAL """
      all_features = tf.keras.layers.concatenate(encoded_features)
      x = tf.keras.layers.Dense(256, activation="relu")(all_features)
      x = tf.keras.layers.Dropout(0.5)(x)
      output = tf.keras.layers.Dense(cantidad_posibles_predicciones)(x)

      model = tf.keras.Model(all_inputs, output)


      model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=["accuracy"])


      """ Gráfico de conectividad """
      tf.keras.utils.plot_model(model, to_file='./frontend/public/files/Connectivity_graph.png', show_shapes=True, rankdir="LR")


      """ ENTRENAMIENTO DEL MODELO """
      early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

      history = model.fit(train_ds, epochs=200, validation_data=val_ds, verbose=0, callbacks=[early_stop])
      #history = model.fit(train_ds, epochs=1, validation_data=val_ds, verbose=1, callbacks=[early_stop])

      ### Gráficos del entrenamiento
      from io import StringIO

      fig = plt.figure()
      plt.plot(history.history['accuracy'])
      plt.plot(history.history['val_accuracy'])
      #plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['Training', 'Validation'], loc='upper left')
      #plt.show()

      # Código para exportar en SVG
      imgdata = StringIO()
      fig.savefig(imgdata, format='svg')
      imgdata.seek(0)
      graph1 = imgdata.getvalue()

      fig = plt.figure()
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      #plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['Training', 'Validation'], loc='upper left')
      #plt.show()

      # SVG
      imgdata = StringIO()
      fig.savefig(imgdata, format='svg')
      imgdata.seek(0)
      graph2 = imgdata.getvalue()


      # Visualización del progreso de entrenamiento del modelo usando las estadísticas almacenadas en el objeto history.
      hist = pd.DataFrame(history.history)
      hist['epoch'] = history.epoch
      dataframe_hist_tail = hist.tail() # Almacena los resultados en una variable para enviarlos al frontend
      hist.tail()

      # Transforma el Dataframe con los resultados del entrenamiento a json para enviarlos al frontend
      datos_hist_tail = json.loads(dataframe_hist_tail.to_json(orient='records'))


      # Resultados del entrenamiento con datos de testeo
      loss, accuracy = model.evaluate(test_ds)
      print("Accuracy", accuracy)
      accuracy = accuracy * 100


      """ Sección de Prueba """
      # Quita la columna a predecir (target) del dataset de testeo y finalmente almacena los datos target en una lista
      test_labels = test.pop('code_target')

      # Se realiza la predicción en todos los datos de pruebas
      predicciones = model.predict(test_ds)

      predicciones_number = []
      for i, logits in enumerate(predicciones):
        class_idx = tf.argmax(logits).numpy()
        predicciones_number.append(class_idx)


      # Se almacenan las predicciones en un Dataframe
      predicciones_df = pd.DataFrame(predicciones_number, columns = ['Predicciones'])

      # Se Almacena los resultados originales en un Dataframe
      test_labels_df = pd.DataFrame(test_labels.to_list(), columns = ['Original'])

      # Unión de los Dataframe creados anteriormente
      df_originals_predictions = test_labels_df.join(predicciones_df)

      # Convierte el código asignado al texto original 
      for index, row in df_originals_predictions.iterrows():
        df_originals_predictions['Original'][index] = class_names[row['Original']]
        df_originals_predictions['Predicciones'][index] = class_names[row['Predicciones']]


      """ Matrix de confusión """
      from sklearn.metrics import confusion_matrix
      matrix = confusion_matrix(df_originals_predictions['Original'].to_list(),df_originals_predictions['Predicciones'].to_list(), labels=class_names)
      # Transforma la matrix de confusión en un dataframe
      df_matrix = pd.DataFrame(matrix, columns=class_names, index=class_names)

      fig = plt.figure()
      matrix_graph = sns.heatmap(df_matrix, annot=True, cbar=False, cmap='Blues')
      matrix_graph.set(xlabel='Predicted',ylabel='Original')
      matrix_graph.set_yticklabels(class_names, rotation=0, va="center")

      # Código para exportar en SVG
      imgdata = StringIO()
      fig.savefig(imgdata, format='svg')
      imgdata.seek(0)
      graph3 = imgdata.getvalue()


      # Se crea el index del Dataframe anterior para una mejor visualización en el frontend
      # Crea la columna index
      df_originals_predictions[''] = range(1, len(df_originals_predictions) + 1)
      # Se ubica la última columna en la primera posición
      cols = list(df_originals_predictions)
      cols = [cols[-1]] + cols[:-1]
      df_originals_predictions = df_originals_predictions[cols]

      # Transforma el Dataframe a json para enviarlos al frontend
      df_originals_predictions_json = json.loads(df_originals_predictions.to_json(orient='records'))


      """ Guardando la Red Neuronal """
      model.save('modelo_red_neuronal')

      # Transforma la carpeta del modelo neuronal en un zip para poder descargarla por html
      import shutil
      shutil.make_archive('modelo_red_neuronal', 'zip', 'modelo_red_neuronal')

      # Mueve el archivo.zip a las carpetas de react, fue la única forma de evitar que la página se actualizara
      from pathlib import Path
      Path("modelo_red_neuronal.zip").rename("./frontend/public/files/modelo_red_neuronal.zip")


      """ Se envían los datos al Frontend """
      data = [{
          'target_predecir_after_train': request.data["target_predecir"], # Target: valor a predecir por la Red Neuronal
          'columns_numeric': columns_numeric, # Todas las variables utilizadas (sin fechas y sin el target)
          'columnas_categoricas_onehot_json': columnas_categoricas_onehot_json, # Columnas categóricas con sus respectivas colunas unique()
          'len_train_dataset': len_train_dataset, # Se utiliza para hacer la pila con la partición de los datos
          'len_val_dataset': len_val_dataset, # Se utiliza para hacer la pila con la partición de los datos
          'len_test_dataset': len_test_dataset, # Se utiliza para hacer la pila con la partición de los datos
          'grafico1': graph1, # Gráfico 1 del entrenamiento
          'grafico2': graph2, # Gráfico 2 del entrenamiento
          'grafico3': graph3, # matrix de confusión
          'datos_hist_tail': datos_hist_tail, # Datos del entrenamiento para ser visualizados en una tabla
          'loss': loss, # Datos del entrenamiento
          'mae': accuracy, # Datos del entrenamiento
          'df_originals_predictions_json': df_originals_predictions_json, # Datos del target original vs predicciones, en el frontend se visualizan en una tabla
          'tipo_red_neuronal': 'string', # Tipo de Red Neuronal
          'class_names': class_names, # Almacena lo que significa para código en una predicción string
      }]

  return Response(data)

# Se consulta la red neuronal
@api_view(['GET', 'POST'])
def consult_red_neuronal(request):
  import pathlib
  import matplotlib # Necesario para exportar en svg
  matplotlib.use('Agg') # Necesario para exportar en svg
  import matplotlib.pyplot as plt
  import pandas as pd
  import seaborn as sns

  import tensorflow as tf

  from tensorflow import keras
  from tensorflow.keras import layers
  import numpy as np

  if(request.data["red_neuronal_type"] == 'number'):
    # Función que se utiliza para reconocer los string o number que vienen del form (el json trae todos los datos en string)
    def is_number(n):
      try:
          float(n)   # Type-casting the string to `float`.
                    # If string is not a valid `float`, 
                    # it'll raise `ValueError` exception
      except ValueError:
          return False
      return True

    # Almacena los datos que vienen del form para la predicción en una variable
    datos_consulta = request.data['data_consult_red_neuronal']

    # Se transforman los números del json que vienen en string a float
    for name_column, value_column in datos_consulta.items():
      if(is_number(datos_consulta[name_column]) == True):
        datos_consulta[name_column] = float(value_column)


    """ Carga el Modelo de la Red Neuronal """
    reloaded_model = tf.keras.models.load_model('modelo_red_neuronal')


    """ Se realiza la predicción """
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in datos_consulta.items()}
    predictions = reloaded_model.predict(input_dict)
    result_predict = predictions[0][0]
    print(result_predict)


    """ Se envían los datos al frontend """
    data = [{
      'result_predict': result_predict
    }]

  if(request.data["red_neuronal_type"] == 'string'):
    class_names = request.data['class_names']

    # Función que se utiliza para reconocer los string o number que vienen del form (el json trae todos los datos en string)
    def is_number(n):
      try:
          float(n)   # Type-casting the string to `float`.
                    # If string is not a valid `float`, 
                    # it'll raise `ValueError` exception
      except ValueError:
          return False
      return True

    # Almacena los datos que vienen del form para la predicción en una variable
    datos_consulta = request.data['data_consult_red_neuronal']

    # Se transforman los números del json que vienen en string a float
    for name_column, value_column in datos_consulta.items():
      if(is_number(datos_consulta[name_column]) == True):
        datos_consulta[name_column] = float(value_column)


    """ Carga el Modelo de la Red Neuronal """
    reloaded_model = tf.keras.models.load_model('modelo_red_neuronal')


    """ Se realiza la predicción """
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in datos_consulta.items()}
    predictions = reloaded_model.predict(input_dict)

    for i, logits in enumerate(predictions):
      class_idx = tf.argmax(logits).numpy()
      p = tf.nn.softmax(logits)[class_idx].numpy() * 100
      name = class_names[class_idx]

      result_predict = name
      porcentaje_predict = int(p)
      print('result_predict: ' , result_predict)
      print('porcentaje_predict: ' , porcentaje_predict)


    """ Se envían los datos al frontend """
    data = [{
      'result_predict': result_predict,
      'porcentaje_predict': porcentaje_predict
    }]

  if(request.data["red_neuronal_type"] == 'binary'):

    print(request.data['class_names'])
    class_names = request.data['class_names']

    # Función que se utiliza para reconocer los string o number que vienen del form (el json trae todos los datos en string)
    def is_number(n):
      try:
          float(n)   # Type-casting the string to `float`.
                    # If string is not a valid `float`, 
                    # it'll raise `ValueError` exception
      except ValueError:
          return False
      return True

    # Almacena los datos que vienen del form para la predicción en una variable
    datos_consulta = request.data['data_consult_red_neuronal']

    # Se transforman los números del json que vienen en string a float
    for name_column, value_column in datos_consulta.items():
      if(is_number(datos_consulta[name_column]) == True):
        datos_consulta[name_column] = float(value_column)


    """ Carga el Modelo de la Red Neuronal """
    reloaded_model = tf.keras.models.load_model('modelo_red_neuronal')


    """ Se realiza la predicción """
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in datos_consulta.items()}
    predictions = reloaded_model.predict(input_dict)

    prob = tf.nn.sigmoid(predictions[0])
    number = prob.numpy().round().astype(int) # Número de la predicción

    result_predict = class_names[number[0]] # El número de la predición se lleva a texto
    porcentaje_predict = int(prob.numpy()[0] * 100)  # Porcentaje de acierto

    print('result_predict: ' , result_predict)
    print('porcentaje_predict: ' , porcentaje_predict)


    """ Se envían los datos al frontend """
    data = [{
      'result_predict': result_predict,
      'porcentaje_predict': porcentaje_predict
    }]

  return Response(data)