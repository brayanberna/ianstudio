import './App.css';
import React, { useState} from 'react';
import ReactPaginate from 'react-paginate';
import ReactFlow, { MiniMap, addEdge, removeElements, Controls, Background } from 'react-flow-renderer';

class App extends React.Component {
  constructor(props){
    super(props);
    this.state = {
      todoList:[],
      red_neuronal_list:[],
      dir_excel:null,
      datos_excel:[],
      columns_excel:[],
      columns_excel_sin_datetime:[],
      activeItem:{
        id:null,
        title:'',
        completed:false,
      },
      editing:false,
      count:0,  //Contador utilizado para reconocer la posición de visualización de datos
      numero_paginas:0, //Número de paginas según el total de datos y cantidad de datos por páginas
      filtered:[], //Datos filtrados separados por la cant de datos a visualizar
      busquedadTableExcel:'',
      pageNumber:0, //deja la page 1 activada
      columna_seleccionada: '', // Columna selecciona por un form select permitiendo visualizar la columna target

      datos_entrenamiento: [],
      count_predict_original:0,  //Contador utilizado para reconocer la posición de visualización de datos
      nro_pag_total_predict_original:0, //Número de paginas según el total de datos y cantidad de datos por páginas
      nro_pag_actual_predict_original:0, //deja la page 1 activada

      /* Estructura de la red neuronal */
      hidden_cap1: 0,
      hidden_cap2: 0,
      hidden_cap3: 0,
      hidden_cap4: 0,
      hidden_cap5: 0,
      hidden_cap6: 0,
      len_train_dataset: '',
      len_val_dataset: '',
      len_test_dataset: '',
      red_neuronal_type: '',
      class_names: [], // ej: 1: iris setosa, 2:iris versicolor
      target_predecir_after_train: '',  //Se guarda el target predicho luego del entrenamiento neuronal
      consult_red_neuronal: '',  //Se guardan para enviar los datos para predecir
      result_predict: '',
      porcentaje_predict: 0,
      pre_loading_train: 0,
      loagind_training: 0,
      post_loading_training: 0,
      elements_public: [],
    }

    this.fetchTasks = this.fetchTasks.bind(this)
    this.handleChange = this.handleChange.bind(this)
    this.handleSubmit = this.handleSubmit.bind(this)
    this.getCookie = this.getCookie.bind(this)

    this.startEdit = this.startEdit.bind(this)
    this.deleteItem = this.deleteItem.bind(this)
    this.strikeUnstrike = this.strikeUnstrike.bind(this)

    this.run_red_neuronal = this.run_red_neuronal.bind(this)
    this.handleChange_neural = this.handleChange_neural.bind(this)
    this.handleSubmit_neural = this.handleSubmit_neural.bind(this)

    this.buscadorTableExcel = this.buscadorTableExcel.bind(this)
    this.handlePageClick = this.handlePageClick.bind(this)

    this.selectColumnTable = this.selectColumnTable.bind(this)
    this.clickPageTablePredictOriginals = this.clickPageTablePredictOriginals.bind(this)
    this.modeloPredeterminadoRedNeuronal = this.modeloPredeterminadoRedNeuronal.bind(this)

    this.consult_red_handleChange = this.consult_red_handleChange.bind(this)
    this.consult_red_neuronal = this.consult_red_neuronal.bind(this)
  };
  
  /* csrftoken Ruta = 'https://docs.djangoproject.com/en/3.2/ref/csrf/' */
  getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
  }

  //Activa funciones
  componentWillMount(){
    this.fetchTasks()
  }

  run_red_neuronal(e){
    this.setState({
      loagind_training: 1,
      post_loading_training: 0,
    })
    e.preventDefault()
    /* Limpia los input y selects del form de predicciones */
    if (document.getElementsByClassName("input").length > 0){
      for(var i=0; i<document.getElementsByClassName("input").length; i++){
        document.getElementsByClassName("input")[i].value = "";
      }
    }
    var csrftoken = this.getCookie('csrftoken')
    /* var url = 'http://127.0.0.1:8000/run_red_neuronal/' */
    var url = 'https://ianstudio.herokuapp.com/run_red_neuronal/'

    fetch(url, {
      method:'POST',
      headers:{
        'Content-type':'application/json',
        'X-CSRFToken':csrftoken,
      },
      body:JSON.stringify({
        target_predecir: this.state.columna_seleccionada,
        datos: this.state.datos_excel,
      })
    }).then(response => response.json()) 
      .then(data => 
        this.setState({
          loagind_training: 2,
          result_predict: '', /* Limpia el resultado de la predicción */
          porcentaje_predict: 0,
          datos_entrenamiento:data,
          nro_pag_total_predict_original: Math.ceil(data[0].df_originals_predictions_json.length/5),
          target_predecir_after_train: data[0].target_predecir_after_train,
          len_train_dataset: data[0].len_train_dataset,
          len_val_dataset: data[0].len_val_dataset,
          len_test_dataset: data[0].len_test_dataset,
          red_neuronal_type: data[0].tipo_red_neuronal,
          class_names: data[0].class_names
        }, () => console.log(""), console.log("data[0].class_names"), console.log())
      )
  }

  // Listando Tasks
  fetchTasks(){
    /* fetch('http://127.0.0.1:8000/task-list/') */
    fetch('https://ianstudio.herokuapp.com/task-list/')
    .then(response => response.json())
    .then(data =>
      this.setState({
        todoList:data
      }, () => {
        console.log("")
      })
    )
  }

  handleChange_neural(e){
    /* if (document.getElementsByClassName("react-flow").length > 0){
      document.querySelector('.react-flow__renderer svg').remove('svg')
      console.log("")
      setTimeout(() => {  console.log("World!"); }, 5000);
    } */

    this.setState({
      dir_excel: e.target.files[0]
    })
  }

  //Agregando
  handleChange(e){
    var name = e.target.name
    var value = e.target.value
    console.log('Name:', name)
    console.log('Value:', value)

    this.setState({
      activeItem:{
        ...this.state.activeItem,
        title:value
      }
    })
  }

  //Botón de enviar
  handleSubmit_neural(e){
    this.setState({
      pre_loading_train: 1,
      elements_public: [],
      columna_seleccionada: '',
      hidden_cap1: 0,
      loagind_training: 0,
      post_loading_training: 0,
    })

    /* Limpia el input del buscador de la tabla con los datos */
    if (document.getElementsByClassName("search_table_excel").length > 0){
      document.getElementsByClassName("search_table_excel")[0].value = "";
    }
    /* Limpia el select del target */
    if (document.getElementsByClassName("form_select_target").length > 0){
      document.getElementsByClassName("form_select_target")[0].value = "";
    }

    e.preventDefault()
    let formData = new FormData();
    formData.append('dir_excel',this.state.dir_excel)

    var csrftoken = this.getCookie('csrftoken')
    /* var url = 'http://127.0.0.1:8000/load_data/' */
    var url = 'https://ianstudio.herokuapp.com/load_data/'
    fetch(url, {
      method:'POST',
      headers:{
        Accept: 'application/json, text/plain, */*',
      },
      body:formData,
    }).then(response => response.json())
      .then(data =>
        this.setState({
          pre_loading_train: 2,
          count: 0, /* Al cargar un nuevo dataset vuelve a la pag. 1 */
          pageNumber: 0, /* Al cargar un nuevo dataset vuelve a la pag. 1 */
          busquedadTableExcel:'', /* Limpia el buscador al cargar un nuevo dataset */
          datos_excel:data[0].datos_json,
          columns_excel:data[0].datos_json[0],
          columns_excel_sin_datetime:data[0].dataframe_sin_datetime[0],
          /* numero_paginas: Math.ceil(data.length/5) */
          numero_paginas: Math.ceil(data[0].datos_json.length/5)
        }, () => {this.modeloPredeterminadoRedNeuronal()}
        )
      )
  }

  //Botón de enviar
  handleSubmit(e){
    e.preventDefault()
    
    /* csrftoken */
    var csrftoken = this.getCookie('csrftoken')

    /* var url = 'http://127.0.0.1:8000/task-create/' */
    var url = 'https://ianstudio.herokuapp.com/task-create/'

    if(this.state.editing == true){  /* EDITANDO */
      /* url = `http://127.0.0.1:8000/task-update/${this.state.activeItem.id}/` */
      url = `https://ianstudio.herokuapp.com//task-update/${this.state.activeItem.id}/`
      this.setState({
        editing:false
      })
    }
    
    fetch(url, {
      method:'POST',
      headers:{
        'Content-type':'application/json',
        /* csrftoken */
        'X-CSRFToken':csrftoken,
      },
      body:JSON.stringify(this.state.activeItem)
    }).then((response) => {
        this.fetchTasks()
        this.setState({
          activeItem:{
          id:null,
          title:'',
          completed:false,
        }
        })
    }).catch(function(error){
      console.log('ERROR:', error)
    })
  }

  /* EDITAR */
  startEdit(task){
    this.setState({
      activeItem:task,
      editing:true,
    })
  }

  /* ELIMINAR */
  deleteItem(task){
    var csrftoken = this.getCookie('csrftoken')

    /* fetch(`http://127.0.0.1:8000/task-delete/${task.id}/`, { */
    fetch(`https://ianstudio.herokuapp.com/task-delete/${task.id}/`, {
      method:'DELETE',
      headers:{
        'Content-type':'application/json',
        'X-CSRFToken':csrftoken,
      },
    }).then((response) => {
      this.fetchTasks()
    })
  }

  /* Subrayado */
  strikeUnstrike(task){
    task.completed = !task.completed
    var csrftoken = this.getCookie('csrftoken')
    /* var url = `http://127.0.0.1:8000/task-update/${task.id}/` */
    var url = `https://ianstudio.herokuapp.com/task-update/${task.id}/`
      fetch(url, {
        method:'POST',
        headers:{
          'Content-type':'application/json',
          'X-CSRFToken':csrftoken,
        },
        body:JSON.stringify({'completed': task.completed, 'title': task.title})
      }).then(() =>{
        this.fetchTasks()
      })
    console.log("")
  }

  /* Función de Input Type File */
  btn_input_file_active() {
    document.querySelector("#ruta_excel").click()
    document.querySelector("#ruta_excel").addEventListener("change", function(){
      if(this.value){
        let valueStore = this.value.split('\\');
        document.querySelector(".file_name").textContent = valueStore[valueStore.length-1]
      }
      else {
        document.querySelector(".file_name").textContent = "Ningún archivo seleccionado"
      }
    })
  }

  /* Filtro para la tabla */
  /* filteredDataExcel(prevProps, prevState) { */
  filteredDataExcel() {
    if(this.state.busquedadTableExcel.length === 0){
      return [this.state.datos_excel.slice(this.state.count, this.state.count + 5), Math.ceil(this.state.datos_excel.length/5)]
    }

    const filtered = this.state.datos_excel.filter(item => {
      return Object.keys(item).some(key => 
        item[key].toString().toLowerCase().includes(this.state.busquedadTableExcel.toLowerCase())
      );
    });

    //const filtered = this.state.datos_excel.filter( poke => poke.Equipo.toLowerCase().includes(this.state.busquedadTableExcel.toLowerCase()))
    return [filtered.slice(this.state.count, this.state.count + 5), Math.ceil(filtered.length/5)]
  }

  buscadorTableExcel(e) {
    this.setState({
      count: 0,
      pageNumber: 0,
      busquedadTableExcel: e.target.value,
    }, () => {
      this.setState({numero_paginas: this.filteredDataExcel()[1]})
      this.colorselectColumnTable()
    })
  }

  handlePageClick(e) {
    this.setState({
      count: e.selected * 5,
      pageNumber: e.selected
    }, () => {
      this.colorselectColumnTable()
    })
  }

  selectColumnTable(e) {

    if(e.target.value == ''){
      this.setState({
        columna_seleccionada: '',
        elements_public: [] }, () => {
        this.colorselectColumnTable();
        this.modeloPredeterminadoRedNeuronal();
      });
    }
    else {
      this.setState({
        columna_seleccionada: e.target.value,
        elements_public: [] }, () => {
        this.colorselectColumnTable();
      });


      e.preventDefault()
      var csrftoken = this.getCookie('csrftoken')
      /* var url = 'http://127.0.0.1:8000/detect_type_neuronal/' */
      var url = 'https://ianstudio.herokuapp.com/detect_type_neuronal/'
      fetch(url, {
        method:'POST',
        headers:{
          'Content-type':'application/json',
          'X-CSRFToken':csrftoken,
        },
        body:JSON.stringify({
          target_predecir: e.target.value,
          datos: this.state.datos_excel,
        })
      }).then(response => response.json()) 
        .then(data => 
          this.setState({
            red_neuronal_type: data[0].tipo_red_neuronal,
            hidden_cap1: data[0].hidden_cap1
          }, () => { this.modeloPredeterminadoRedNeuronal(); })
      )
    }
  }

  colorselectColumnTable() {
    /* Elimina la columna seleccionada anteriormente (Remueve la clase activeColorTable) */
    var td = document.createElement('td');
    td.classList.remove("activeColorTable")

    document.querySelectorAll(".activeColorTable").forEach(function(element) {
      element.classList.remove("activeColorTable");
    });

    /* Selecciona la columna indica por form select */
    var element = document.getElementsByClassName(this.state.columna_seleccionada)
    for(var i = 0; i < element.length; i++)
    {
      element[i].className += " activeColorTable";
    }
  }

  filteredDataOriginalsPredict() {
    return this.state.datos_entrenamiento[0].df_originals_predictions_json.slice(this.state.count_predict_original, this.state.count_predict_original + 5)
  }

  clickPageTablePredictOriginals(e) {
    this.setState({
      count_predict_original: e.selected * 5,
      nro_pag_actual_predict_original: e.selected
    })
  }

  /* Librería reactflow*/
  modeloPredeterminadoRedNeuronal(){
    const onLoad = (reactFlowInstance) => reactFlowInstance.fitView();

    let elements = []
    elements.push({ id: 'input_layer', sourcePosition: 'right', type: 'input', data: { label: 'Input Layer' }, position: { x: 50, y: 0 } })
    elements.push({ id: 'hidden_layer', sourcePosition: 'right', type: 'input', data: { label: 'Hidden Layers' }, position: { x: 300, y: 0 } })
    elements.push({ id: 'output_layer', sourcePosition: 'right', type: 'input', data: { label: 'Output Layer' }, position: { x: 550, y: 0 } })
    elements.push({ id: 'total_neurons_output_layer', sourcePosition: 'right', type: 'input', data: { label: '1 neurona' }, position: { x: 550, y: 42 } })

    /* Se crea el array que contiene las neuronas de la input layer  */
    const elements_input_layer = []
    Object.keys(this.state.columns_excel_sin_datetime).map((key, i) => {
      return ([
        elements_input_layer.push(key)
      ])
    })

    /* Elimina la columna target del array de la input layer*/
    if(this.state.columna_seleccionada != ''){
      elements_input_layer.splice(elements_input_layer.indexOf(this.state.columna_seleccionada), 1);
    }

    /* Agrega como título la cantidad total de neuronas en la capa Input Layer */
    elements.push({ id: 'total_neurons_input_layer', sourcePosition: 'right', type: 'input', data: { label: elements_input_layer.length + ' neuronas' }, position: { x: 50, y: 42 } },)

    /* Agrega las neuronas en la Input Layer */
    var position = [80] /* Constante para ubicar las neuronas en la capa, como coordenadas gps */
    elements_input_layer.map((key, i) => {
      return ([
        elements.push({ id: key, sourcePosition: 'right', type: 'input', data: { label: key }, position: { x: 50, y: position[0] } },),
        position.push(position[0] + 80), /* position = [80, 160] */
        position.splice(0, 1), /* position = [160] # elimina la primera posición*/
      ])
    })


    /* Agrega el título de la cantidad total de neuronas de la primera hidden layers*/
    elements.push({ id: 'total_neurons_hidden_layer', sourcePosition: 'right', type: 'input', data: { label: this.state.hidden_cap1 + ' neuronas' }, position: { x: 300, y: 42 } },)

    /* Creación de las neuronas de la hidden layers 1 */
    /* Unión entre las neuronas de la hidden layers 1 e Input Layer*/
    if(this.state.hidden_cap1 !== 0){
      if(this.state.hidden_cap1 <= 8){
        for (var i = this.state.hidden_cap1; i > 0; i--){
          /* Crea las neuronas */
          elements.push({ id: 'capaOculta1_neurona' + i, targetPosition: 'left', sourcePosition: 'right', data: { label: <div>Neurona {i}</div> }, position: { x: 300, y: 80 * i } })
          /* Agrega las conexiones entre la input layer y la primera hidden layer */
          elements_input_layer.map((neurona) => {
            elements.push({ id: 'e'+ i, source: neurona, target: 'capaOculta1_neurona' + i, animated: true , style: { stroke: '#5da7d6' } })
          })
        }
      }
      else {
        let cant_total_neuronas = this.state.hidden_cap1
        for (var i = this.state.hidden_cap1; i > 0; i--){
          if(i == cant_total_neuronas){
            elements.push({ id: 'capaOculta1_neurona' + i, targetPosition: 'left', sourcePosition: 'right', data: { label: <div>Neurona {i}</div> }, position: { x: 300, y: 640 } })
            elements.push({ id: 'punto_seguido1', className: 'punto_seguido',  data: { label: <div></div> }, position: { x: 300, y: 550 } })
            elements.push({ id: 'punto_seguido2', className: 'punto_seguido', data: { label: <div></div> }, position: { x: 300, y: 570 } })
            elements.push({ id: 'punto_seguido3', className: 'punto_seguido', data: { label: <div></div> }, position: { x: 300, y: 590 } })
            elements_input_layer.map((neurona) => {
              elements.push({ id: 'e3' + i, source: neurona, target: 'capaOculta1_neurona' + i, animated: true , style: { stroke: '#5da7d6' } })
            })
          }
          if(i <=6){
            /* Crea las neuronas */
            elements.push({ id: 'capaOculta1_neurona' + i, targetPosition: 'left', sourcePosition: 'right', data: { label: <div>Neurona {i}</div> }, position: { x: 300, y: 80 * i } })
            /* Agrega las conexiones entre la input layer y la primera hidden layer */
            elements_input_layer.map((neurona) => {
              elements.push({ id: 'e3' + i, source: neurona, target: 'capaOculta1_neurona' + i, animated: true , style: { stroke: '#5da7d6' } })
            })
          }
        }
      }
    }
    
    /* Agrega la neurona de la capa Output Layer */
    if(this.state.columna_seleccionada != ''){
      elements.push({ id: this.state.columna_seleccionada, targetPosition: 'left', type: 'output', data: { label: <div>{this.state.columna_seleccionada}</div> }, position: { x: 550, y: position[0] / 2 } })
    }

    /* Unión (conexión) entre las neuronas de la hidden layer y Output layer */
    for (var i = 6; i > 0; i--){
      if(this.state['hidden_cap'+i] !== 0){
      /* if('this.state.hidden_cap'+i !== 0){ */
        for (var neu = 1; neu <= this.state['hidden_cap'+i]; neu++){
          /* console.log('capaOculta' + i + '_neurona' + neu) */
          if (neu < 7 || neu == this.state['hidden_cap'+i]) {
            elements.push({ id: 'e4' + i + neu, source: 'capaOculta' + i + '_neurona' + neu, target: this.state.columna_seleccionada, animated: true , style: { stroke: '#5da7d6' } })
          }
        }
      }
    }

    this.setState({
      elements_public: elements
    })

    return elements
  }

  consult_red_handleChange(e) {
    if(e.target.name == 'select'){
      const value = e.target.value
      const name = e.target.value

      this.setState({
        consult_red_neuronal: {
          ...this.state.consult_red_neuronal,
          [name]: value
        }
      })
    }
    else {
      const value = e.target.value
      const name = e.target.name

      this.setState({
        consult_red_neuronal: {
          ...this.state.consult_red_neuronal,
          [name]: value
        }
      })
    }
  }

  /* Consultar la Red Neuronal */
  consult_red_neuronal(e){
    this.setState({
      post_loading_training: 1
    })
    console.log(this.state.consult_red_neuronal)
    e.preventDefault()
    var csrftoken = this.getCookie('csrftoken')
    /* var url = 'http://127.0.0.1:8000/consult_red_neuronal/' */
    var url = 'https://ianstudio.herokuapp.com/consult_red_neuronal/'
    fetch(url, {
      method:'POST',
      headers:{
        'Content-type':'application/json',
        'X-CSRFToken':csrftoken,
      },
      body: JSON.stringify({
        data_consult_red_neuronal:this.state.consult_red_neuronal,
        red_neuronal_type: this.state.red_neuronal_type,
        class_names: this.state.class_names,
      })
    }).then(response => response.json()) 
      .then(data => 
        this.setState({
          post_loading_training: 2,
          result_predict: data[0].result_predict,
          porcentaje_predict: data[0].porcentaje_predict
        }, () => console.log(data[0].result_predict), console.log("RESULTADO PREDICT: "))
      )
  }


  render(){
    var tasks = this.state.todoList
    var datos_excel = this.state.datos_excel
    var columnas_excel = this.state.columns_excel
    var columnas_excel_sin_datetime = this.state.columns_excel_sin_datetime
    var self = this
    return(
      <div className="container">
        {/* Selección de dataset */}
        <div className="background_one" style={{ 'background-image': 'url(../files/bg_light_blue_right.png)'}}>
          <h1 className="title_principal text-gradient-blue-purple">Seleccione una&nbsp;<strong>fuente de datos</strong>&nbsp;para iniciar el proceso {/* <img  src={process.env.PUBLIC_URL + '/files/data-cleaning.svg'}></img> */}</h1>
          <div className="button_load_excel_grid">
            <form onSubmit={this.handleSubmit_neural}>
              <input onChange={this.handleChange_neural} type="file" className="input_file_display" id="ruta_excel" name="ruta_excel"></input>
              <div className="button_load_excel_item">
                <div className="btn_input_file" onClick={this.btn_input_file_active} style= {{ 'background-image': 'url(../files/cubes_red.svg)'}}>
                  <p className="icon_input_file">
                    <i className='bx bx-upload'></i>
                    <p className="file_name">Ningún archivo seleccionado</p>
                  </p>
                  <p className="title_input_file">Seleccionar Archivo</p>
                  <p className="text_input_file">Cargue archivos excel, archivos csv o archivos de texto.</p>
                </div>
                <div className="item_btn_submit">
                  <input className="btn_submit_file" id="submit" type="submit" name="Add" value="Cargar Datos"/>
                </div>
              </div>
            </form>
          </div>
        </div>

        {/* LOADING... */}
        { this.state.pre_loading_train === 1  &&
            <div className='contenedor_spinner'>
              <div className='spinner'></div>
            </div>
        }

        {/* Table con visualización de datos */}
        { this.state.pre_loading_train === 2  &&
          <div className="table_excel_grid">
            <div className="table_excel_item">
              <div className="structure_table">
                <h3 className="title_table">Visualización del conjunto de datos</h3>
                <div className="btn_search_table_excel"> 
                  <i className='bx bx-search-alt-2' ></i>
                  <input onChange={this.buscadorTableExcel} type="text" placeholder="Buscador" className="search_table_excel"></input>
                </div>
                <table id="customers">
                  <tr>
                    {Object.keys(columnas_excel).map(key => (
                        <th className={key}>{key}</th>
                      ))}
                  </tr>
                  {
                    this.filteredDataExcel()[0].map(function(list_neural) {
                      return (
                        <tr>
                          {
                            Object.keys(columnas_excel).map((key, i) => (
                              <td className={key}>{list_neural[key]}</td>
                              ))
                            }
                        </tr>
                      )
                    })
                  }
                </table>
                <div className="footer_table_grid">
                  <div className="dataTables_info">
                    <p>{datos_excel.length} filas × {Object.keys(columnas_excel).length - 1} columns</p>
                  </div>
                  <div className="multipagination">
                    <div>
                    <ReactPaginate
                      previousLabel={<i className='bx bx-chevron-left'></i>}
                      nextLabel={<i className='bx bx-chevron-right'></i>}
                      breakLabel={"..."}
                      breakClassName={"break-me"}
                      pageCount={this.state.numero_paginas}
                      marginPagesDisplayed={2}
                      pageRangeDisplayed={5}
                      forcePage={this.state.pageNumber}
                      onPageChange={this.handlePageClick}
                      containerClassName={"pagination"}
                      subContainerClassName={"pages pagination"}
                      activeClassName={"active"}>
                    </ReactPaginate>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        }

        { this.state.pre_loading_train === 2  &&
          <form onSubmit={this.run_red_neuronal}>
            {/* Seleccionar target*/}
            <div className="select_target_grid" style= {{ 'padding-top': ''}}>
              <div className="form_select_item">
                <img className="img_target" src={process.env.PUBLIC_URL + '/files/target2.png'}></img>
                <h4>Seleccionar Columna a <strong style={{ 'background-image': 'url(../files/decoration_title.svg)'}}>Predecir</strong></h4>
                <div className="form_select_content">
                  <select onChange={this.selectColumnTable} className="form_select_target" aria-label="Default select example" required>
                      <option selected value="">Seleccionar</option>
                      {Object.keys(columnas_excel_sin_datetime).filter(key => key !== '').map(key => (
                        <option className={key}>{key}</option>
                      ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Ejecutar red neuronal con btn (RUN) */}
                <div className="run_red_neuronal_grid" style= {{ 'padding-top': ''}}>
                  <div className="run_red_neuronal_item">
                    <button className="run_red_neuronal_btn_blue_submit" id="submit" type="submit" name="Add" value="">
                      Entrenar Red Neuronal <i class='bx bx-play-circle'></i>
                    </button>
                  </div>
                </div>
          </form>
        }

        {/* Estructura de la Red Neuronal */}
        { this.state.pre_loading_train === 2  &&
          <div className="design_neural_network">
            <h1>Configuración predeterminada del modelo <img  src={process.env.PUBLIC_URL + '/files/training_result2.png'}></img></h1>
              <ReactFlow 
                onLoad={(reactFlowInstance) => reactFlowInstance.fitView()}
                /* elements={this.modeloPredeterminadoRedNeuronal()} */
                elements={this.state.elements_public}
                defaultZoom={0.78}
                defaultPosition={[200, 8]}
                style={{width:'1007px', height:'60vh'}}
                snapToGrid={true}
                snapGrid={[5, 5]}
                zoomOnScroll={false}
                zoomOnPinch={false}
              >
                <Controls/>
                <Background 
                  variant="lines"
                />
                <addEdge/>
              </ReactFlow>
          </div>
        }

        {/* LOADING... */}
        { this.state.loagind_training === 1  &&
          <div className='contenedor_spinner'>
            <div className='spinner'></div>
          </div>
        }

        {/* Gráfico de barra de porcentajes sobre datos de entrenamiento y validación */}
        { this.state.loagind_training === 2  &&
          <div className="bar_divide_bar_wrapper">
            <div className="bar_divide_bar_header_img">
              <img  src={process.env.PUBLIC_URL + '/files/train_vali_test2.svg'}></img>
              {/* <object  data={process.env.PUBLIC_URL + '/files/train_vali_test2.svg'}></object> */}
            </div>
            <div className="bar_divide_bar_header">
              <h2 className="center">División del DataFrame en conjuntos de entrenamiento, validación y prueba</h2>
  {/*             <div className="background_icon_bar_divide">
                <i class='bx bx-cut'></i>
              </div> */}
            </div>
            <div className="bar_divide_bar_container">
                    <div className="training_set">
                      <span className="training_set_cantidad">{this.state.len_train_dataset} ejemplos</span>
                      <span className="training_set_porcentaje">80%</span>
                    </div>
                    <div className="val_set">
                      <span className="val_set_cantidad">{this.state.len_val_dataset}</span>
                      <span className="val_set_porcentaje">10%</span>
                    </div>
                    <div className="test_set">
                      <span className="test_set_cantidad">{this.state.len_test_dataset}</span>
                      <span className="test_set_porcentaje">10%</span>
                    </div>
                    <span className="subconjuntos training_set_conj">Training Set</span>
                    <span className="subconjuntos val_set_conj">Val Set</span>
                    <span className="subconjuntos test_set_conj">Test Set</span>
            </div>
          </div>
        }

        {/* Gráfico de conectividad */}
        { this.state.loagind_training === 2  &&
          this.state.datos_entrenamiento.map(function(list_neural, index){
            return (
              <React.Fragment>
                <div className="graficos_connectivity_grid">
                  <div className="graf_connectivity_img">
                    <img  src={process.env.PUBLIC_URL + '/files/connectivity2.png'}></img>
                    {/* <object  data={process.env.PUBLIC_URL + '/files/connectivity3.svg'}></object> */}
                  </div>
                  <div className="graficos_connectivity_header">
                    <h2 className="grafico_connectivity_title">Arquitectura del modelo resultante (gráfico de conectividad)</h2>
                  </div>
                  <a href={process.env.PUBLIC_URL + '/files/Connectivity_graph.png'} target="_blank">
                    {/* <img src={require('./images/Connectivity_graph.png').default}></img> */}
                    <img src={process.env.PUBLIC_URL + '/files/Connectivity_graph.png'} alt="Image1"></img>

                    {/* <img src="http://localhost:8000/frontend/public/files/Connectivity_graph.png" alt="Image1"></img> */}

                    {/* <img src="http://localhost:8000/files/Connectivity_graph.png" alt="Image1" ></img> */}
                    {/* <img src="%PUBLIC_URL%/Connectivity_graph.png" alt="Image1" ></img> */}
                    {/* <img src="http://localhost:8000/Connectivity_graph.png" alt="Image1" ></img> */}
                  </a>
                </div>
              </React.Fragment>
            )
          })
        }

        {/* Gráficos de entrenamiento (precisión y perdida)*/}
        { this.state.loagind_training === 2  &&
          this.state.datos_entrenamiento.map(function(list_neural, index){
            if(self.state.red_neuronal_type != '')
              return(
                <div className="graficos_training_wrapper">
                    <div className="graficos_training_img">
                      <img  src={process.env.PUBLIC_URL + '/files/training_result3.png'}></img>
                    </div>
                    <div className="graficos_training_header">
                      {/* <div className="background_icon_graph_training">
                        <i class='bx bx-pulse'></i>
                      </div> */}
                      <h2 className="graficos_training_title">Resultados del entrenamiento de la red neuronal en los conjuntos de entrenamiento y validación</h2>
                    </div>
                  <div className="graficos_abs_square_grid">
                    <div className="graph_precision">
                      <h2>Visualice la precisión del modelo a lo largo del tiempo</h2>
                      <div className="img_graph_precision" dangerouslySetInnerHTML={{__html: list_neural.grafico1}} />
                    </div>
                    <div className="graph_perdida">
                      <h2>Visualice la función de pérdida a lo largo del tiempo</h2>
                      <div className="img_graph_perdida" dangerouslySetInnerHTML={{__html: list_neural.grafico2}} />
                    </div>

                    {/* Tabla con estadisticas finales del entrenamiento */}
                    <div className="table_excel_grid3">
                      <div className="table_statics_jupyter_item">
                        <div className="structure_table_statics_jupyter">
                          <h3 className="title_table_statics_jupyter">Rendimiento general del modelo</h3>
                          <table id="statics_jupyter">
                            <tr>
                            {Object.keys(list_neural.datos_hist_tail[0]).map(key => (
                                  <th className={key}>{key}</th>
                            ))}
                            </tr>
                            {
                              list_neural.datos_hist_tail.map(function(list_neural_statics) {
                                return (
                                  <tr>
                                    {
                                      Object.keys(list_neural.datos_hist_tail[0]).map((key, i) => (
                                        <td className={key}>{ list_neural_statics[key] % 1 != 0 ? (parseFloat(list_neural_statics[key]).toFixed(4)): list_neural_statics[key]}</td>
                                        ))
                                      }
                                  </tr>
                                )
                              })
                            }
                          </table>
                        </div>
                      </div>
                    </div>

                  </div>
                </div>
              )
          })
        }

        {/* Resultados de loss, mse y mae*/}
        { this.state.loagind_training === 2  &&
          this.state.datos_entrenamiento.map(function(list_neural, index){
            if(self.state.red_neuronal_type === 'number')
              return(
                <div className="statics_container">
                  <img className="img_target" src={process.env.PUBLIC_URL + '/files/timer.svg'}></img>
                  <h2 className="section_title">Evaluación del modelo entrenado en el conjunto de datos de prueba <img  src={process.env.PUBLIC_URL + '/files/acurracy.svg'}></img></h2>
                  <div className="statics_row">
                    <div className="statics_grid loss_result">
                      <div className="statics_number">{ parseFloat(list_neural.loss).toFixed(2) }</div>
                      <p className="statics_sub_title">loss</p>
                    </div>
                    <div className="statics_grid accuracy_result">
                      <div className="statics_number">{ parseFloat(list_neural.mae).toFixed(2) }</div>
                      <p className="statics_sub_title">mae</p>
                    </div>
                    <div className="statics_grid mse_result">
                      <div className="statics_number">{ parseFloat(list_neural.mse).toFixed(2) }</div>
                      <p className="statics_sub_title">mse</p>
                    </div>
                  </div>
                </div>
              )
            
            if(self.state.red_neuronal_type === 'string' || self.state.red_neuronal_type === 'binary')
              return(
                <div className="statics_container">
                  <img className="img_target" src={process.env.PUBLIC_URL + '/files/timer.svg'}></img>
                  <h2 className="section_title">Evaluación del modelo entrenado en el conjunto de datos de prueba <img  src={process.env.PUBLIC_URL + '/files/acurracy.svg'}></img></h2>
                  <div className="statics_row_Accuracy">
                    <div className="statics_grid loss_result">
                      <div className="statics_number">{ parseFloat(list_neural.loss).toFixed(0) }</div>
                      <p className="statics_sub_title">loss</p>
                    </div>
                    <div className="statics_grid accuracy_result">
                      <div className="statics_number">{ parseFloat(list_neural.mae).toFixed(0) }%</div>
                      <p className="statics_sub_title">Accuracy </p>
                    </div>
                  </div>
                </div>
              )
          })
        }

        {/* Table con result originales vs predicciones - Matriz de confusión*/}
        { this.state.loagind_training === 2  &&
          this.state.red_neuronal_type === 'string'  &&
            <div className="table_result_and_matris_wrapper">
              <div className="table_excel_grid2">
                {/* Tabla con predicciones y datos originales */}
                <div className="table_excel_item2">
                  {this.state.datos_entrenamiento.map(function(data_train, index){
                    return(
                      <div className="structure_table">
                        <h3 className="title_table">Predicciones en el conjunto de prueba</h3>
                        <table id="customers">
                          <tr>
                            {Object.keys(data_train.df_originals_predictions_json[0]).map(key => (
                              <th className={key}>{key}</th>
                            ))}
                          </tr>

                          {
                            self.filteredDataOriginalsPredict().map(function(list_neural) {
                              if(self.state.red_neuronal_type === 'number')
                                return (
                                  <tr>
                                    {
                                      Object.keys(data_train.df_originals_predictions_json[0]).map((key, i) => (
                                        <td className={key}>{ key == '' ? (list_neural[key]): parseFloat(list_neural[key]).toFixed(2)}</td>
                                      ))
                                    }
                                  </tr>
                                )
                              if(self.state.red_neuronal_type === 'string' || self.state.red_neuronal_type === 'binary')
                                return (
                                  <tr>
                                    {
                                      Object.keys(data_train.df_originals_predictions_json[0]).map((key, i) => (
                                        <td className={key}>{ key == '' ? (list_neural[key]): list_neural[key]}</td>
                                      ))
                                    }
                                  </tr>
                                )
                            })
                          }
                        </table>
                        <div className="footer_table_grid">
                          <div className="dataTables_info">
                            <p>{data_train.df_originals_predictions_json.length} filas</p>
                          </div>
                          <div className="multipagination">
                            <div>
                            <ReactPaginate
                              previousLabel={<i className='bx bx-chevron-left'></i>}
                              nextLabel={<i className='bx bx-chevron-right'></i>}
                              breakLabel={"..."}
                              breakClassName={"break-me"}
                              pageCount={self.state.nro_pag_total_predict_original}
                              marginPagesDisplayed={2}
                              pageRangeDisplayed={5}
                              forcePage={self.state.nro_pag_actual_predict_original}
                              onPageChange={self.clickPageTablePredictOriginals}
                              containerClassName={"pagination"}
                              subContainerClassName={"pages pagination"}
                              activeClassName={"active"}>
                            </ReactPaginate>
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Matriz de confusión */}
                {this.state.datos_entrenamiento.map(function(list_neural, index){
                  if(self.state.red_neuronal_type != '')
                    return(
                      <div className="graph_matriz_confusion">
                        <h2>Matriz de confusión de los datos originales vs. las predicciones</h2>
                        <div className="img_graph_confusion" dangerouslySetInnerHTML={{__html: list_neural.grafico3}} />
                      </div>
                    )
                })}
              </div>
            </div>
        }

        { this.state.loagind_training === 2  &&
          this.state.red_neuronal_type === 'binary'  &&
            <div className="table_result_and_matris_wrapper">
              <div className="table_excel_grid2">
                {/* Tabla con predicciones y datos originales */}
                <div className="table_excel_item2">
                  {this.state.datos_entrenamiento.map(function(data_train, index){
                    return(
                      <div className="structure_table">
                        <h3 className="title_table">Predicciones en el conjunto de prueba</h3>
                        <table id="customers">
                          <tr>
                            {Object.keys(data_train.df_originals_predictions_json[0]).map(key => (
                              <th className={key}>{key}</th>
                            ))}
                          </tr>

                          {
                            self.filteredDataOriginalsPredict().map(function(list_neural) {
                              if(self.state.red_neuronal_type === 'number')
                                return (
                                  <tr>
                                    {
                                      Object.keys(data_train.df_originals_predictions_json[0]).map((key, i) => (
                                        <td className={key}>{ key == '' ? (list_neural[key]): parseFloat(list_neural[key]).toFixed(2)}</td>
                                      ))
                                    }
                                  </tr>
                                )
                              if(self.state.red_neuronal_type === 'string' || self.state.red_neuronal_type === 'binary')
                                return (
                                  <tr>
                                    {
                                      Object.keys(data_train.df_originals_predictions_json[0]).map((key, i) => (
                                        <td className={key}>{ key == '' ? (list_neural[key]): list_neural[key]}</td>
                                      ))
                                    }
                                  </tr>
                                )
                            })
                          }
                        </table>
                        <div className="footer_table_grid2">
                          <div className="dataTables_info">
                            <p>{data_train.df_originals_predictions_json.length} filas</p>
                          </div>
                          <div className="multipagination">
                            <div>
                            <ReactPaginate
                              previousLabel={<i className='bx bx-chevron-left'></i>}
                              nextLabel={<i className='bx bx-chevron-right'></i>}
                              breakLabel={"..."}
                              breakClassName={"break-me"}
                              pageCount={self.state.nro_pag_total_predict_original}
                              marginPagesDisplayed={2}
                              pageRangeDisplayed={5}
                              forcePage={self.state.nro_pag_actual_predict_original}
                              onPageChange={self.clickPageTablePredictOriginals}
                              containerClassName={"pagination"}
                              subContainerClassName={"pages pagination"}
                              activeClassName={"active"}>
                            </ReactPaginate>
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Matriz de confusión */}
                {this.state.datos_entrenamiento.map(function(list_neural, index){
                  if(self.state.red_neuronal_type != '')
                    return(
                      <div className="graph_matriz_confusion">
                        <h2>Matriz de confusión de los datos originales vs. las predicciones</h2>
                        <div className="img_graph_confusion" dangerouslySetInnerHTML={{__html: list_neural.grafico3}} />
                      </div>
                    )
                })}
              </div>
            </div>
        }

        { this.state.loagind_training === 2  &&
          this.state.red_neuronal_type === 'number'  &&
            <div className="table_result_and_matris_wrapper">
              <div className="table_excel_grid4">
                {/* Tabla con predicciones y datos originales */}
                <div className="table_excel_item3">
                  {this.state.datos_entrenamiento.map(function(data_train, index){
                    return(
                      <div className="structure_table">
                        <h3 className="title_table">Predicciones en el conjunto de prueba</h3>
                        <table id="customers">
                          <tr>
                            {Object.keys(data_train.df_originals_predictions_json[0]).map(key => (
                              <th className={key}>{key}</th>
                            ))}
                          </tr>

                          {
                            self.filteredDataOriginalsPredict().map(function(list_neural) {
                              if(self.state.red_neuronal_type === 'number')
                                return (
                                  <tr>
                                    {
                                      Object.keys(data_train.df_originals_predictions_json[0]).map((key, i) => (
                                        <td className={key}>{ key == '' ? (list_neural[key]): parseFloat(list_neural[key]).toFixed(2)}</td>
                                      ))
                                    }
                                  </tr>
                                )
                              if(self.state.red_neuronal_type === 'string' || self.state.red_neuronal_type === 'binary')
                                return (
                                  <tr>
                                    {
                                      Object.keys(data_train.df_originals_predictions_json[0]).map((key, i) => (
                                        <td className={key}>{ key == '' ? (list_neural[key]): list_neural[key]}</td>
                                      ))
                                    }
                                  </tr>
                                )
                            })
                          }
                        </table>
                        <div className="footer_table_grid2">
                          <div className="dataTables_info">
                            <p>{data_train.df_originals_predictions_json.length} filas</p>
                          </div>
                          <div className="multipagination">
                            <div>
                            <ReactPaginate
                              previousLabel={<i className='bx bx-chevron-left'></i>}
                              nextLabel={<i className='bx bx-chevron-right'></i>}
                              breakLabel={"..."}
                              breakClassName={"break-me"}
                              pageCount={self.state.nro_pag_total_predict_original}
                              marginPagesDisplayed={2}
                              pageRangeDisplayed={5}
                              forcePage={self.state.nro_pag_actual_predict_original}
                              onPageChange={self.clickPageTablePredictOriginals}
                              containerClassName={"pagination"}
                              subContainerClassName={"pages pagination"}
                              activeClassName={"active"}>
                            </ReactPaginate>
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
        }

        {/* Formulario de la Red Neuronal */}
        { this.state.loagind_training === 2  &&
          <form onSubmit={this.consult_red_neuronal}>
            <div className="consultas_red_neuronal_wrapper" /* style= {{ 'background-image': 'url(../files/redes.png)'}} */>
              <div className="consultas_red_neuronal_header">
                <div className="background_icon">
                  {/* <i class='bx bx-git-repo-forked'></i> */}
                  <img  src={process.env.PUBLIC_URL + '/files/studio.svg'}></img>
                  {/* <object  data={process.env.PUBLIC_URL + '/files/machine-learning.svg'}></object> */}
                </div>
                <h2 className="center">Utilice el modelo para hacer predicciones sobre nuevos datos</h2>
              </div>
              <div className="consultas_red_neuronal_container">
                {
                  this.state.datos_entrenamiento.map((list) => (
                          Object.keys(list.columns_numeric).map((number_column) => {
                              return (
                                <div>
                                  <label className="label_input" for={list.columns_numeric[number_column]}>{list.columns_numeric[number_column].replace('_', ' ')}:</label>
                                  <input className="input" type="number" step="0.001" keyboardType='numeric' id={list.columns_numeric[number_column]} 
                                  name={list.columns_numeric[number_column]}
                                  onChange={this.consult_red_handleChange} required/>
                                </div>
                              )
                          })
                  ))
                }


                {
                  this.state.datos_entrenamiento.map((list) => (
                    Object.keys(list.columnas_categoricas_onehot_json).map((name_column, i) => (

                      <div className="">
                          <label className="label_input">Seleccionar {name_column}:</label>
                          <select onChange={this.consult_red_handleChange} className="input" aria-label="Default select example" name={name_column} required>
                            <option selected value="">Seleccionar</option>
                          {
                            list.columnas_categoricas_onehot_json[name_column].map((name_column2) => {
                              return (
                                <option value={ name_column2 } name={{ name_column2 }}>{ name_column2.replace(name_column + '_', '') }</option>
                              )
                            })
                          }
                          </select>
                      </div>
                    ))
                  ))
                }
              </div>
              <div className="consultas_red_neuronal_submit">
                <button className="btn_submit_blue_xl" type="submit">
                  Predecir
                  {/* <i class='bx bx-git-repo-forked'></i> */}
                  {/* <i class='bx bx-brain'></i> */}
                  <i class='bx bxs-brain'></i>
                </button>
              </div>
            </div>
          </form>
        }

        {/* LOADING... */}
        { this.state.post_loading_training === 1  &&
          <div className='contenedor_spinner'>
            <div className='spinner'></div>
          </div>
        }

        {/* Visualización del Resultado de la Predicción */}
        { this.state.post_loading_training === 2  &&
          <div className="result_consult_red_neuronal_wrapper">
            <h2 className="title_consult_red_neuronal center">Predicción<i class='bx bxl-redux'></i></h2>
            {
              this.state.red_neuronal_type === 'number' &&
                <div className="result_consult_red_neuronal_container">
                  <h1>{parseFloat(this.state.result_predict).toFixed(3)}</h1>
                  <span>{this.state.target_predecir_after_train}</span>
                </div>
            }
            {
              this.state.red_neuronal_type === 'string'  &&
                <div className="result_consult_red_neuronal_container">
                  <h1>{this.state.porcentaje_predict}%</h1>
                  <h2>{this.state.result_predict}</h2>
                  <span>{this.state.target_predecir_after_train}</span>
                </div>
            }
            {
              this.state.red_neuronal_type === 'binary' &&
                <div className="result_consult_red_neuronal_container">
                  <h1>{this.state.porcentaje_predict}%</h1>
                  <h2>{this.state.result_predict}</h2>
                  <span>{this.state.target_predecir_after_train}</span>
                </div>
            }
          </div>
        }

        {/* Descargar el Modelo de la Red Neuronal */}
        { this.state.loagind_training === 2  &&
          <div className="run_red_neuronal_grid" style= {{ 'padding-top': ''}}>
            <div className="run_red_neuronal_item">
              <div className="item_btn_submit">
                {/* <a href="http://localhost:8000/frontend/public/files/modelo_red_neuronal.zip" download="red_neuronal_model" */}
                <a href={process.env.PUBLIC_URL + '/files/modelo_red_neuronal.zip' } download="red_neuronal_model"
                className="btn_download_model">
                  <i class='bx bxs-download'></i>
                  Descargar Modelo Entrenado
                </a>
              </div>
            </div>
          </div>
        }

        {/* Explicación de como consultar el modelo de la red neuronal */}
        { this.state.loagind_training === 2  &&
          this.state.red_neuronal_type === 'string'  &&
            <div className="terminal_consult_red_neuronal_wrapper">
              <div className="terminal_consult_red_neuronal_header">
                <div className="circulored"></div>
                <div className="circuloyellow"></div>
                <div className="circulocalipso"></div>
                <h2 className="terminal_consult_red_neuronal_title">Utilice el modelo entrenado para hacer predicciones en python</h2>
              </div>
              <div className="terminal_consult_red_neuronal_row_grid">
                  <p className="index">1</p>
                  <p><span className="igual">import</span> tensorflow <span className="igual">as</span> tf</p>

                  <p className="index">2</p>
                  <p className="comentario"># Etiquetas representadas en nombres</p>

                  <p className="index">3</p>
                  <p>class_names <span className="igual">=</span> [<span className="texto">'</span>
                      {
                        <span className="texto">{this.state.class_names.join("', '")}</span>
                      }
                    <span className="texto">'</span>]
                  </p>

                  <p className="index">4</p>
                  <p className="comentario"># Cargar el modelo</p>

                  <p className="index">5</p>
                  <p>model <span className="igual">=</span> tf.<span className="tensorflow">keras</span>.<span className="tensorflow">models</span>.<span className="tensorflow">load_model</span>(<span className="texto">'C:/Users/56975/Downloads/modelo_red_neuronal'</span>)</p>

                  <p className="index">6</p>
                  <p className="comentario"># Datos necesarios para la predicción</p>

                  <p className="index">7</p>
                  <p>sample <span className="igual">=</span> &#123; </p>
                    {
                      this.state.datos_entrenamiento.map((list) => (
                        Object.keys(list.columnas_categoricas_onehot_json).map((name_column, index) => (
                          <React.Fragment>
                            <p className="index"> </p>
                            <p>&emsp; <span className="texto">'{name_column}'</span>: <span className="string">'String'</span>,</p>
                          </React.Fragment>
                        ))
                      ))
                    }
                    {
                      this.state.datos_entrenamiento.map((list) => (
                              Object.keys(list.columns_numeric).map((number_column, index) => (
                                <React.Fragment>
                                  <p className="index"> </p>
                                  <p>&emsp; <span className="texto">'{list.columns_numeric[number_column]}'</span>: <span className="number">Number</span>,</p>
                                </React.Fragment>
                              ))
                      ))
                    }

                  <p className="index">8</p>
                  <p>&#125;</p>

                  <p className="index">9</p>
                  <p>input_dict <span className="igual">=</span> &#123;name: tf.<span className="tensorflow">convert_to_tensor</span>([value]) <span className="for">for</span> name, value <span className="for">in</span> sample.<span className="tensorflow">items</span>()&#125;</p>

                  <p className="index">10</p>
                  <p className="comentario"># Llamar al método y predecir</p>

                  <p className="index">11</p>
                  <p>predictions <span className="igual">=</span> model.<span className="tensorflow">predict</span>(input_dict)</p>

                  <p className="index">12</p>
                  <p><span className="for">for</span> i, logits <span className="for">in</span> <span className="enumerate">enumerate</span>(predictions):</p>
                    <p className="index">13</p>
                    <p>&emsp; class_idx <span className="igual">=</span> tf.<span className="tensorflow">argmax</span>(logits).<span className="tensorflow">numpy</span>()</p>
                    <p className="index">14</p>
                    <p>&emsp; p <span className="igual">=</span> tf.<span className="tensorflow">nn</span>.<span className="tensorflow">softmax</span>(logits)[class_idx].<span className="tensorflow">numpy</span>() <span className="for">*</span> <span className="number">100</span></p>
                    <p className="index">15</p>
                    <p>&emsp; result_predict <span className="igual">=</span> class_names[class_idx]</p>
                    <p className="index">16</p>
                    <p>&emsp; percentage_predict <span className="igual">=</span> <span className="type">int</span>(p)</p>
              </div>
            </div>
        }

        { this.state.loagind_training === 2  &&
          this.state.red_neuronal_type === 'binary'  &&
            <div className="terminal_consult_red_neuronal_wrapper">
              <div className="terminal_consult_red_neuronal_header">
                <div className="circulored"></div>
                <div className="circuloyellow"></div>
                <div className="circulocalipso"></div>
                <h2 className="terminal_consult_red_neuronal_title">Utilice el modelo entrenado para hacer predicciones en python</h2>
              </div>
              <div className="terminal_consult_red_neuronal_row_grid">
                  <p className="index">1</p>
                  <p><span className="igual">import</span> tensorflow <span className="igual">as</span> tf</p>

                  <p className="index">2</p>
                  <p className="comentario"># Etiquetas representadas en nombres</p>

                  <p className="index">3</p>
                  <p>class_names <span className="igual">=</span> [<span className="texto">'</span>
                      {
                        <span className="texto">{this.state.class_names.join("', '")}</span>
                      }
                    <span className="texto">'</span>]
                  </p>

                  <p className="index">4</p>
                  <p className="comentario"># Cargar el modelo</p>

                  <p className="index">5</p>
                  <p>model <span className="igual">=</span> tf.<span className="tensorflow">keras</span>.<span className="tensorflow">models</span>.<span className="tensorflow">load_model</span>(<span className="texto">'C:/Users/56975/Downloads/modelo_red_neuronal'</span>)</p>

                  <p className="index">6</p>
                  <p className="comentario"># Datos necesarios para la predicción</p>

                  <p className="index">7</p>
                  <p>sample <span className="igual">=</span> &#123; </p>
                    {
                      this.state.datos_entrenamiento.map((list) => (
                        Object.keys(list.columnas_categoricas_onehot_json).map((name_column, index) => (
                          <React.Fragment>
                            <p className="index"> </p>
                            <p>&emsp; <span className="texto">'{name_column}'</span>: <span className="string">'String'</span>,</p>
                          </React.Fragment>
                        ))
                      ))
                    }
                    {
                      this.state.datos_entrenamiento.map((list) => (
                              Object.keys(list.columns_numeric).map((number_column, index) => (
                                <React.Fragment>
                                  <p className="index"> </p>
                                  <p>&emsp; <span className="texto">'{list.columns_numeric[number_column]}'</span>: <span className="number">Number</span>,</p>
                                </React.Fragment>
                              ))
                      ))
                    }

                  <p className="index">8</p>
                  <p>&#125;</p>

                  <p className="index">9</p>
                  <p>input_dict <span className="igual">=</span> &#123;name: tf.<span className="tensorflow">convert_to_tensor</span>([value]) <span className="for">for</span> name, value <span className="for">in</span> sample.<span className="tensorflow">items</span>()&#125;</p>

                  <p className="index">10</p>
                  <p className="comentario"># Llamar al método y predecir</p>

                  <p className="index">11</p>
                  <p>predictions <span className="igual">=</span> model.<span className="tensorflow">predict</span>(input_dict)</p>

                  <p className="index">12</p>
                  <p>prob <span className="igual">=</span> tf.<span className="tensorflow">nn</span>.<span className="tensorflow">sigmoid</span>(predictions[<span className="number">0</span>])</p>

                  <p className="index">13</p>
                  <p>number <span className="igual">=</span> prob.<span className="tensorflow">numpy</span>().<span className="tensorflow">round</span>().<span className="tensorflow">astype</span>(int)</p>

                  <p className="index">14</p>
                  <p>result_predict <span className="igual">=</span> class_names[number[<span className="number">0</span>]]</p>

                  <p className="index">15</p>
                  <p>percentage_predict <span className="igual">=</span> <span className="type">int</span>(prob.<span className="tensorflow">numpy</span>()[<span className="number">0</span>] * <span className="number">100</span>)</p>
              </div>
            </div>
        }

        { this.state.loagind_training === 2  &&
          this.state.red_neuronal_type === 'number'  &&
            <div className="terminal_consult_red_neuronal_wrapper">
              <div className="terminal_consult_red_neuronal_header">
                <div className="circulored"></div>
                <div className="circuloyellow"></div>
                <div className="circulocalipso"></div>
                <h2 className="terminal_consult_red_neuronal_title">Utilice el modelo entrenado para hacer predicciones en python</h2>
              </div>
              <div className="terminal_consult_red_neuronal_row_grid">
                  <p className="index">1</p>
                  <p><span className="igual">import</span> tensorflow <span className="igual">as</span> tf</p>

                  <p className="index">2</p>
                  <p className="comentario"># Cargar el modelo</p>

                  <p className="index">3</p>
                  <p>model <span className="igual">=</span> tf.<span className="tensorflow">keras</span>.<span className="tensorflow">models</span>.<span className="tensorflow">load_model</span>(<span className="texto">'C:/Users/56975/Downloads/modelo_red_neuronal'</span>)</p>

                  <p className="index">4</p>
                  <p className="comentario"># Datos necesarios para la predicción</p>

                  <p className="index">5</p>
                  <p>sample <span className="igual">=</span> &#123; </p>
                    {
                      this.state.datos_entrenamiento.map((list) => (
                        Object.keys(list.columnas_categoricas_onehot_json).map((name_column, index) => (
                          <React.Fragment>
                            <p className="index"> </p>
                            <p>&emsp; <span className="texto">'{name_column}'</span>: <span className="string">'String'</span>,</p>
                          </React.Fragment>
                        ))
                      ))
                    }
                    {
                      this.state.datos_entrenamiento.map((list) => (
                              Object.keys(list.columns_numeric).map((number_column, index) => (
                                <React.Fragment>
                                  <p className="index"> </p>
                                  <p>&emsp; <span className="texto">'{list.columns_numeric[number_column]}'</span>: <span className="number">Number</span>,</p>
                                </React.Fragment>
                              ))
                      ))
                    }

                  <p className="index">6</p>
                  <p>&#125;</p>

                  <p className="index">7</p>
                  <p>input_dict <span className="igual">=</span> &#123;name: tf.<span className="tensorflow">convert_to_tensor</span>([value]) <span className="for">for</span> name, value <span className="for">in</span> sample.<span className="tensorflow">items</span>()&#125;</p>

                  <p className="index">8</p>
                  <p className="comentario"># Llamar al método y predecir</p>

                  <p className="index">9</p>
                  <p>predictions <span className="igual">=</span> model.<span className="tensorflow">predict</span>(input_dict)</p>

                  <p className="index">10</p>
                  <p>result_predict = predictions[<span className="number">0</span>][<span className="number">0</span>]</p>
              </div>
            </div>
        }




        <div id="task-container" style= {{ 'display': 'none'}}>
          <div id="form-wrapper">
            <form onSubmit={this.handleSubmit} id="form">
              <div className="flex-wrapper">
                <div style={{flex:6}}>
                  <input onChange={this.handleChange} className="form-control" id="title" value={this.state.activeItem.title} type="text" name="title" placeholder="Add task.." />
                </div>

                <div style={{flex:1}}>
                  <input className="btn btn-warning" id="submit" type="submit" name="Add" />
                </div>
              </div>
            </form>
          </div>

          <div id="list-wrapper">
            {tasks.map(function(tasks, index){
              return(
                <div key={index} className="task-wrapper flex-wrapper">
                  <div onClick={() => self.strikeUnstrike(tasks)} style={{flex:7}}>

                    {tasks.completed == false ? (
                      <span>{tasks.title}</span>
                    ) : (
                      <strike>{tasks.title}</strike>
                    )}

                  </div>
                  <div style={{flex:1}}>
                    <button onClick={() => self.startEdit(tasks)} className="btn btn-sm btn-outline-info">Edit</button>
                  </div>

                  <div style={{flex:1}}>
                  <button onClick={() => self.deleteItem(tasks)} className="btn btn-sm btn-outline-info">-</button>
                  </div>
                </div>
              )
            })}
          </div>

        </div>
      </div>
    )
  }
}

export default App;