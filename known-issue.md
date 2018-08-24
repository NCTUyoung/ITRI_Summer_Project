---
description: Some issue due to the frontend and backend I use. There will be  pro and con.
---

# Known Issue

## Not able to extend to large web application

Plotly Dash is python web framework based on flask . Since the model visualization is appropriate  for small web app, so I use single web file to develop.  
Multi page application is also possible  check out [here](https://dash.plot.ly/urls)  
but there seem to be some issue on GPU usage on flask multi process usage .  

So it is not able to dynamic load model after web app is running, you have to load it first and start the process.

## RuntimeError: main thread is not in main loop

Due to matplot default backend will call tinker for gui rendering.Since we don't use matplot gui interface.So, change backend to 'Agg' solve the problem  
[http://matplotlib.1069221.n5.nabble.com/Matplotlib-Tk-and-multithreading-td40647.html](http://matplotlib.1069221.n5.nabble.com/Matplotlib-Tk-and-multithreading-td40647.html)

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

## How to clear all selection in loss dataframe?

In current version there is no button to clear all selection.The only way is to deselect them alternative.Or, you can select all and deselect all.

