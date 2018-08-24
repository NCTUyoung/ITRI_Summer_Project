---
description: Some frequently question and useful tip are collect here
---

# FAQ

## HTML template support 

Dash do not official support template rendering.To use template, you need to install`dash_dangerously_set_inner_html`

and use in this way. I use this method to render each card in dynamic image pool.You can check the code for detail usage. 

```python
template = """
        <link rel="import" href="./assets/facets-jupyter.html"></link>
        <h>hello</h>
        <facets-dive id="hello" height="800" sprite-image-width="28" sprite-image-height="28" atlas-url="./assets/atlas.jpg"></facets-dive>
        """
template = template.format(jsonstr=jsonstr)

html.Div([
    dash_dangerously_set_inner_html.DangerouslySetInnerHTML(template)
])
```

## How to add new  dataset ?

For the time being , you can only select mnist and cifar10 as dataset.  
But, you can write parser to load your data into `(x_train, y_train), (x_test, y_test)`



## How to change style of web app?

css file in the `assets` directory will automatic add to web , load the stylesheet and add className into html code

## How to add javascript file into dash?

`grasia_dash_components` have some extend component for dash , please install it first.  
Then,  in the python you can use import component as below code.

```python
import grasia_dash_components as gdc

# add the conponent into html block
gdc.Import(src="./assets/test.js")
```

## How to show numpy array to dash web app?

1. First you need to turn array into base64 string.
2. Send the template string to `Img` tag

```python
html.Img(src='data:image/png;base64,{}'.format(data_string),style={'width':'150px'}),
```

If it is a matplot image, save them into buffer,and repeat \(1\)\(2\).

```python
plt.plot(pca_features[i],linewidth=7.0)
buf = io.BytesIO()
plt.savefig(buf, format='JPEG')
buf.seek(0)
im_wave = np.array(Image.open(buf))
im_wave = util.convert_base64(im_wave/255.0) 
```



