---
description: Here you will know how to modify the tool to meet your need
---

# Code Description

## Project Structure

* **root-project-dir**
  * **models**
  * **assets**
  * **utils**
    * util.py
  * **index.py**

\*\*\*\*

* Mode should be place in model dir, there should be `<cifar.h5><mnist.h5>` in it by default.
* Put all external files in assets , this is a static files dir for public access.
* Utils have some predefined helper function.
* Main process is running on index.py

## Code Layout

### Include Library

Because this is a compact visualization tool , many python is being included.Mostly, they are install in the anaconda package.

| Library | Usage |
| :--- | :--- |
| plotly | Main plot library with interactive plot |
| dash | Web app framework for plotly |
| dash\_dangerously\_set\_inner\_html | Dash’s plugin for template rendering |
| grasia\_dash\_components | Dash’s plugin for import external js file |
| dash\_table\_experiments | Dash’s plugin for interactive data table |
| tensorflow | Deep learning framework |

```python
# Tensorflow and keras library
#-----
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from utils import util 
util.restrict_gpu_mem()

# import python library-----
#-----
import io
import glob
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import base64
from textwrap import dedent as d
import json
import pandas as pd
from scipy.signal import savgol_filter

# import plot library
#-----
import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import matplotlib.colors as colors

# import dash library
#-----
import datetime
import dash
from dash.dependencies import Input, Output,Event
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import dash_dangerously_set_inner_html
import grasia_dash_components as gdc

### program setting
model_name = './models/cifar.h5'
dataset_name = 'cifar10'
num_classes=10
```

### Dataset preload

In order to enhance web user experience, preload the model,dataset,feature vector is needed.Some of them are turn into pandas dataframe in the benefit of index tracing.

| Variable | Explain |
| :--- | :--- |
| \(x\_train, y\_train\), \(x\_test, y\_test\) | Preload dataset |
| model, model\_extractfeatures | Preload user model |
| y\_pred, result, loss\_matrix, fc2\_features | Forward result |
| X\_original, X\_embedded, pca\_features | Projection result |
| df\_global, df\_result, df\_predict | Global pandas dataframe |
| loss\_matrix\_light | Matrix use to represent loss alpha color |

```python
# import dataset-----
datasets = tf.keras.datasets

if(dataset_name=='mnist'):
    num_classes=10
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
elif(dataset_name=='cifar10'):
    num_classes=10
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
elif(dataset_name=='cifar100'):
    num_classes =100
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
else:
    raise ValueError('No Dataset selected')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape)

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)



# Load pretrain model-----
print('Loading Model--------------') 
model = tf.keras.models.load_model(model_name)

# Creat Feature Extracter-----
def get_feature_dense(model):
    secend = False
    for layer in reversed(model.layers):
        if 'dense' in layer.name:
            if secend:
                return layer
            secend = True                    
model_extractfeatures = Model(inputs=model.input,
                                 outputs=get_feature_dense(model).output)

# Forward model to get predict and feature vector
y_pred = model.predict(x_train)
result = y_pred.argmax(axis=1)
loss_matrix = np.zeros(len(y_pred))

loss_matrix = util.cross_entropy(y_train,y_pred)
fc2_features = model_extractfeatures.predict(x_train)
print('Finish Loading-------------') 

# Project vector to low dimension to visualize -----
print('Start TSNE/PCA-------------')
sample_num = 500
X_original = TSNE(n_components=2).fit_transform(x_train[0:sample_num].reshape(sample_num,-1))
X_embedded = TSNE(n_components=2).fit_transform(fc2_features[0:sample_num])
# X_original = PCA(n_components=2).fit_transform(x_train.reshape(x_train.shape[0],-1))
pca_features = PCA(n_components=50).fit_transform(fc2_features)
print('Finish TSNE/PCA-------------')



# Global Pandas Dataframe 
df_global = pd.DataFrame(X_embedded)
df_result = pd.DataFrame(result,columns=['Predict'])
df_loss = pd.DataFrame(loss_matrix,columns=['Loss'])
df_label = pd.DataFrame(y_train.argmax(axis=1),columns=['Label'])
df_result = pd.concat([df_result,df_label,df_loss],axis=1)
df_result['index'] = df_result.index
df_predict = pd.DataFrame(util.softmax(y_pred))

# light up the loss to better opacity visualization
loss_matrix_light = np.log2(loss_matrix[:sample_num]/loss_matrix[:sample_num].max()+1)
for i in range(5):
    loss_matrix_light = np.log2(loss_matrix_light+1)

```

### Figure Preload

Some static figure are preload int this block.

```python
# dash app setting -----
app = dash.Dash()
app.config.suppress_callback_exceptions = True
processed_string = ''
processed_string_loss = ''
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'   
})
app.css.append_css({
    'external_url': 'https://www.w3schools.com/w3css/4/w3.css'   
})

# Figure 
#-------------------------
text_hover = np.arange(sample_num)
# Fig 0 -----
data = []
for i in range(10):
    
    data_class0 = X_original[result[0:sample_num]==i]
    if i ==0:
        v = True
    else:
        v = 'legendonly'
    trace = go.Scatter(
        x=data_class0[:,0],
        y=data_class0[:,1],
        mode='markers',
        marker=dict(
            color='rgb'+str(plt.cm.tab10.colors[i]),
            size=8
        ),
        visible = True,
        name = 'Predict class '+str(i),
        hoverinfo = 'text',
        text = 'Predict class '+str(i)
    )   
    data.append(trace)
layout = dict( showlegend=True,hovermode ='closest')
figure0 = dict(data=data, layout=layout)

# Fig 1 -----
data = []
for i in range(10):
    
    data_class0 = X_embedded[result[0:sample_num]==i]
    if i ==0:
        v = True
    else:
        v = 'legendonly'
    trace = go.Scatter(
        x=data_class0[:,0],
        y=data_class0[:,1],
        mode='markers',
        marker=dict(
            color='rgb'+str(plt.cm.tab10.colors[i]),
            size=8
        ),
        visible = True,
        name = 'Predict class '+str(i),
        hoverinfo = 'text',
        text = 'Predict class '+str(i)
    )   
    data.append(trace)
layout = dict( showlegend=True,hovermode ='closest')
figure1 = dict(data=data, layout=layout)

#Fig 2 -----
data = []
for i in range(10):
    
    data_class0 = X_embedded[result[0:sample_num]==i]
    if i ==0:
        v = True
    else:
        v = 'legendonly'
    trace = go.Scatter(
        x=data_class0[:,0],
        y=data_class0[:,1],
        mode='markers',
        marker=dict(
            color='rgb'+str(plt.cm.tab10.colors[i]),
            size=8
        ),
        visible = v,
        name = 'Predict class '+str(i),
        hoverinfo = 'text',
        text = 'Predict class '+str(i)
    )   
    data.append(trace)
layout = dict( showlegend=True,hovermode ='closest')
fig = dict(data=data, layout=layout)


data_loss = []

for i in range(num_classes):
    data_class0 = X_embedded[result[0:sample_num]==i]
    loss_matrix_sample = loss_matrix[:sample_num]
    color = loss_matrix_sample[result[0:sample_num]==i]
    trace = go.Scatter(
        x = data_class0[:,0],
        y = data_class0[:,1],
        mode='markers',
        marker=dict(
            color='rgba'+str(plt.cm.tab10.colors[i]),
            size=8,
            opacity = loss_matrix_light,
            line = dict(
            color = 'rgb(0, 0, 0)',
            width = 1
          )
        ),
        
    )
    data_loss.append(trace)
layout = dict( showlegend=True,hovermode ='closest')
fig_loss = dict(data=data_loss, layout=layout)
#---------------------------------------------------

# Model Summary-------------------------------------
summary = ''  
def get_summary(x):
    global summary 
    summary=summary+x+'\n'
model.summary(print_fn=get_summary)
#---------------------------------------------------


# Dataset Summary-----------------------------------
dataset_length = len(x_train)
```

### HTML Layout

all the static html code are place here.Since I use plot dash , there are some different between naive html, but most of the time dash can achieve the same function.

```python
# Component -----

jsonstr = open('../Facets/jsonstr_mnist.txt','r')
jsonstr = jsonstr.read()

img = Image.open('./assets/atlas.jpg')

code = util.convert_base64(np.array(img))
template = """
        
        <link rel="import" href="./assets/facets-jupyter.html"></link>
        <h>hello</h>
        <facets-dive id="hello" height="800" sprite-image-width="28" sprite-image-height="28" atlas-url="./assets/atlas.jpg"></facets-dive>
        """
template = template.format(jsonstr=jsonstr)
js_code = """
var data = JSON.parse('{jsonstr}');
document.querySelector("#hello").data = data;
    """
js_code = js_code.format(jsonstr=jsonstr)
js_file = open('./assets/test.js','w')
js_file.write(js_code)
js_file.close()

def generate_card(index_list):
    print(index_list)
    card_list = []
    for i in index_list:
        data_string = util.convert_base64(x_train[i]) 
        card_list.append(
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(data_string),style={'width':'100%'}),
                html.Div([
                    html.P('index: {}'.format(df_result.iloc[i]['index'])),
                    html.P('Loss: {:.3e}'.format(df_result.iloc[i]['Loss'])),
                    html.P('Predict: {}'.format(df_result.iloc[i]['Predict'])),
                    html.P('Label: {}'.format(df_result.iloc[i]['Label'])),
                    
                ],className='w3-container text-left bold')
            ],className='w3-card-2',style={'width':'15%'})
        )
    return html.Div(card_list)

# HTML Layout ---------------------------------------
app.layout = html.Div(className='w3-container',children=[
    # title -----
    html.H2(children='ITRI: Model  Visulization',className='text-center'),
    html.Hr(),
    # Section Model Input -----
    html.Div([
        html.Div([
            dcc.Markdown(d("""
                **Model Input**
                
                {}
                
            """.format(model_name))),
            dcc.Markdown(d("""
                **Dataset**

                {}
            """.format(dataset_name))),
            
        ],className='six columns'),
        
        html.Div([
          html.Pre(summary,id='output-model-summary')  
        ],className='six columns'),
        
        
    ],className = 'row'),
    
    # Section Scatter Projection -----
    html.Hr(),
    html.Div(className='row',children=[
        html.Div([
            html.Div('Original Data Projection',className='bold'),
            dcc.Graph(
                id='example-graph',
                figure = figure0
                
            )
        ],className='six columns'),  
        html.Div([
            html.Div('After Training Data Projection',className='bold'),
            dcc.Graph(
                id='example-graph1',
                figure = figure1
            )
        ],className='six columns'),
    ]),
    
    
    # Section Image Click  -----
    html.Hr(),
    html.Div(className='row',children=[
        html.Div([
            dcc.Markdown(d("""
                **Hidden Layer Result Visulization**

                Choose class on the right area.
            """)),
            dcc.Graph(id='fig-click',figure=fig),
        ],className='six columns'),
        html.Div([
            dcc.Markdown(d("""
                **Click Data**

                Click on points in the graph.
            """)),
            html.Pre(id='click-data'),
        ], className='three columns'),
        html.Div([
            dcc.Markdown(d("""
                **Show Image**

                Image Corresponse to data point
            """)),
            html.Img(id='click-img',src=processed_string,className='six columns'),
        ], className='three columns'),
        
    ]),
    
    # Section Loss Scatter -----
    html.Hr(),
    html.Div(className='row',children=[
        html.Div([
            dcc.Markdown(d("""
                **Loss Distribution**
                High Loss data are red.
            """),className='text-center'),
            dcc.Graph(id='fig_loss',figure=fig_loss),
        ],className='six columns'),
        
        html.Div([
            dcc.Markdown(d("""
                **Show Image**

                Image Corresponse to data point
            """)),
            html.Img(id='click-img-loss',src=processed_string_loss,className='six columns'),
        ], className='three columns'),
        
    ]),
    
    # Section Loss Dataframe -----
    html.Hr(),
    html.Div('Loss Dataframe',className='bold text-center'),
    html.Div([
        
        html.Div([
            dt.DataTable(
                # Initialise the rows
                rows=df_result[:sample_num].to_dict('record'),
                row_selectable=True,
                filterable=True,
                sortable=True,
                selected_row_indices=[],
                editable=False,
                id='table'
            ), 
            
        ],className='eight columns'),
        html.Div([
            html.Div('Select Index',className='bold'),
            html.Div(id='selected-indexes')
        ],className='four columns')
        
        
    ],className='row'),
    
    # Image Pool -----
    html.Hr(),
    html.Div([
        html.Div('Image Show Pool',className='bold text-center'),
        html.Div([
            
        ],className='row',id='image-pool')
        
       
        
        
    ],className='row'),
    
    # Image Query -----
    html.Hr(),
    html.Div('Image Query',className='bold text-center'),  
    html.Div([
        html.Div([
            
            html.Br(),
            html.Div('Select index you want to query'),
            dcc.Input(
                placeholder='Enter a value...',
                type='text',
                value='',
                id = 'query-index'
            ),
            html.Div('',id='query-index-info')
            
        ],className='four columns'),
        html.Div([
            html.Img(id='query-image-pool')
        ],className='eight columns'),
        
    ],className='row'),
    
    
    # Facets Dive -----
    html.Hr(),
    html.Div([
        html.Div('Facets Dive',className='bold text-center'),
        html.Div([
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(template)
        ],style={'display':'none'}),
    ],className='row'),
   
    
    
    
    
    # Hidden Conponent To Bind Data -----
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(id='intermediate-value-loss', style={'display': 'none'}),
    html.Div(id='image-pool-click', style={'display': 'none'}),
    gdc.Import(src="./assets/test.js"),
    
    html.Div(id='output-loss-button',style={'display':'none'}),
    

    
])

layout2 = html.Div([
    dash_dangerously_set_inner_html.DangerouslySetInnerHTML(template)
])


```

### Callback

callback define the communication between the html and our python backend.  
[https://dash.plot.ly/getting-started-part-2](https://dash.plot.ly/getting-started-part-2)  
It is worth mention that I use BASE-64 string to render image to frontend. Helper function in util.py`convert_base64` is available to convert numpy array\(float32\) to base64 string.  
Global variable or Hidden dom are use here to communicate between different component.

#### Example, How to bulid 【click and show 】

1. Registered Click event on plotly graph
2. Click event occur , record information into hidden html Dom or global variable
3. Register change event on hidden Dom, when change occur show click data point.

```python
# Callback ------------------------------------------



# click datapoint and image-----
#-----------------------------------
    # global click data
    
@app.callback(Output('intermediate-value', 'children'), [Input('fig-click', 'clickData')])
def Update_click_data(clickData):
    global processed_string
    if(clickData):
        mask = (df_result['Predict'][:sample_num]==clickData['points'][0]['curveNumber']).values.flatten()
        sub_df = df_global.iloc[:sample_num]
        sub_df = sub_df[mask]
        index = sub_df.iloc[clickData['points'][0]['pointIndex']].name
        processed_string = util.convert_base64(x_train[index])
        return json.dumps(clickData, indent=2)

@app.callback(
    Output('click-img', 'src'),
    [Input('intermediate-value', 'children')])
def display_click_data(clickData):
    a = 'data:image/png;base64,{}'.format(processed_string)
    return a

@app.callback(
    Output('click-data', 'children'),
    [Input('intermediate-value', 'children')])
def display_click_data(clickData):
    return clickData



# Loss Scatter Callback-----
@app.callback(Output('intermediate-value-loss', 'children'), [Input('fig_loss', 'clickData')])
def Update_click_loss(clickData):
    global processed_string_loss
    if(clickData):
        mask = (df_result['Predict'][:sample_num]==clickData['points'][0]['curveNumber']).values.flatten()
        sub_df = df_global.iloc[:sample_num]
        sub_df = sub_df[mask]
        index = sub_df.iloc[clickData['points'][0]['pointIndex']].name
        processed_string_loss = util.convert_base64(x_train[index])
        return json.dumps(clickData, indent=2)

@app.callback(
    Output('click-img-loss', 'src'),
    [Input('intermediate-value-loss', 'children')])
def display_click_loss(clickData):
    a = 'data:image/png;base64,{}'.format(processed_string_loss)
    return a


# Loss Dataframe Callback -----
current_index = []
loss_click = 0
@app.callback(
    Output('selected-indexes', 'children'),
    [Input('table', 'rows'),
     Input('table', 'selected_row_indices'),
     
    ])
def update_select_indices(rows, selected_row_indices):
    global current_index
    
    df_temp = pd.DataFrame(rows)
    current_index = df_temp.iloc[selected_row_indices]['index'].values
    if(len(selected_row_indices)>15):
        current_index = current_index[:15]
        return str(current_index)
    return str(current_index)

@app.callback(
    Output('image-pool','children'),
    [Input('selected-indexes','children')]
)
def update_image_pool(selected_row_indices):
    print(current_index)
    return generate_card(current_index)





query_index = 0
query_index_list = []
def get_closest_images(query_image_idx, num_results=5):
    distances =np.linalg.norm(np.subtract(pca_features,pca_features[query_image_idx]),axis=1)
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest


@app.callback(
    Output('query-index-info','children'),
    [Input('query-index','value')]
)
def update_query_index(value):
    global query_index,query_index_list
    try:
        query_index = int(value)
        query_index_list = get_closest_images(query_index)
        return query_index_list
    except:
        return 'invalid input'
    
@app.callback(
    Output('query-image-pool','src'),
    [Input('query-index-info','children')]
)
def update_query_image_pool(index_list):
    if 'invalid' in index_list:
        return ''
    else:
        f,ax = plt.subplots(5,2,figsize=(16,3*5),gridspec_kw = {'width_ratios':[1, 3]})
        for i in range(len(query_index_list)):
            
            feat = fc2_features[query_index_list[i]]
#             feat.max(axis=(0,1,2))
            yhat = savgol_filter(feat, 81, 3) # window size 51, polynomial order 3
            
            
            
            ax[i][0].imshow(x_train[query_index_list[i]])
            ax[i][1].plot(yhat)
            plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='JPEG')
        buf.seek(0)
        im = Image.open(buf)
        
        im = np.array(im)
        buf.close()
        return 'data:image/JPEG;base64,{}'.format(util.convert_base64(im/255.0))
```

## Development Environment Setup

I recommend  use jupyter notebook as IDE.  
You can stop the process, change the html code any time,and resume process.  
This would be like hot reload development.  
`main.ipynb` is file I develop this app.





