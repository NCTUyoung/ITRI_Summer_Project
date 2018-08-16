---
description: Some issue due to the frontend and backend I use. There will be  pro and con.
---

# Known Issue

### Not able to extend to large web application

Plotly Dash is python web framework based on flask . Since the model visualization is appropriate  for small web app, so I use single web file to develop.  
Multi page application is also possible  check out [here](https://dash.plot.ly/urls)  
but there seem to be some issue on GPU usage on flask multi process usage .  

So it is not able to dynamic load model after web app is running, you have to load it first and start the process.





