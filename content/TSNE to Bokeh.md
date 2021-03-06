Title: TSNE to Bokeh Scatterplot
Date: 3/25/17
Category: Workflow
Tags: python, bokeh, tsne
BokehCSS: https://cdn.pydata.org/bokeh/dev/bokeh-0.12.5dev16.min.css
BokehJS: https://cdn.pydata.org/bokeh/dev/bokeh-0.12.5dev16.min.js

This is a workflow I use often in data exploration. TSNE gives a good representation of high-dimensional data, and Bokeh is helpful in creating a simple interactive plots with contextual info given by colors and tooltips. 

This workflow has been extremely helpful for:

- text analytics/NLP tasks if text data is passed through a `TfidfVectorizer` or similar from `scikit-learn`
- understanding `word2vec` or `doc2vec` vectors by passing them to TSNE
- getting an idea of *separability* in doing prediction / classification by passing the outcome variable to bokeh

This example uses the [Australian atheletes data set](http://math.furman.edu/~dcs/courses/math47/R/library/DAAG/html/ais.html), which contains 11 numeric variables. This workflow is even more helpful on larger datsets with higher dimensionality.

### References

> t-Distributed Stochastic Neighbor Embedding (t-SNE) is a (prize-winning) technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets. 

[t-SNE - Laurens van der Maaten](https://lvdmaaten.github.io/tsne/)

> Bokeh is a Python interactive visualization library that targets modern web browsers for presentation.

[Welcome to Bokeh](http://bokeh.pydata.org/en/latest/)

---

```python
from statsmodels.api import datasets
from sklearn.manifold import TSNE
import pandas as pd

from bokeh.plotting import figure, ColumnDataSource, output_notebook, output_file, show, save 
from bokeh.models import HoverTool, WheelZoomTool, PanTool, BoxZoomTool, ResetTool, TapTool, SaveTool
from bokeh.palettes import brewer
output_notebook()
```



<div class="bk-root">
    <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
    <span id="7ad69fb3-1b38-4d81-b231-574c4216b12e">Loading BokehJS ...</span>
</div>





```python
ais = datasets.get_rdataset("ais", "DAAG")
data = ais['data']
```


```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rcc</th>
      <th>wcc</th>
      <th>hc</th>
      <th>hg</th>
      <th>ferr</th>
      <th>bmi</th>
      <th>ssf</th>
      <th>pcBfat</th>
      <th>lbm</th>
      <th>ht</th>
      <th>wt</th>
      <th>sex</th>
      <th>sport</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.96</td>
      <td>7.5</td>
      <td>37.5</td>
      <td>12.3</td>
      <td>60</td>
      <td>20.56</td>
      <td>109.1</td>
      <td>19.75</td>
      <td>63.32</td>
      <td>195.9</td>
      <td>78.9</td>
      <td>f</td>
      <td>B_Ball</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.41</td>
      <td>8.3</td>
      <td>38.2</td>
      <td>12.7</td>
      <td>68</td>
      <td>20.67</td>
      <td>102.8</td>
      <td>21.30</td>
      <td>58.55</td>
      <td>189.7</td>
      <td>74.4</td>
      <td>f</td>
      <td>B_Ball</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.14</td>
      <td>5.0</td>
      <td>36.4</td>
      <td>11.6</td>
      <td>21</td>
      <td>21.86</td>
      <td>104.6</td>
      <td>19.88</td>
      <td>55.36</td>
      <td>177.8</td>
      <td>69.1</td>
      <td>f</td>
      <td>B_Ball</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.11</td>
      <td>5.3</td>
      <td>37.3</td>
      <td>12.6</td>
      <td>69</td>
      <td>21.88</td>
      <td>126.4</td>
      <td>23.66</td>
      <td>57.18</td>
      <td>185.0</td>
      <td>74.9</td>
      <td>f</td>
      <td>B_Ball</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.45</td>
      <td>6.8</td>
      <td>41.5</td>
      <td>14.0</td>
      <td>29</td>
      <td>18.96</td>
      <td>80.3</td>
      <td>17.64</td>
      <td>53.20</td>
      <td>184.6</td>
      <td>64.6</td>
      <td>f</td>
      <td>B_Ball</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_numeric = data.select_dtypes(exclude=['object'])
```


```python
perplexity = 15
learning_rate = 400

tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=666)

tsne_data = tsne.fit_transform(data_numeric)
```

### Formatting data for Bokeh
The easiest/cleanest way to get data into Bokeh is to put everything you'll need (original data, TSNE values, point colorings/other metadata) into a single data frame. You can pass that dataframe to `ColumnDataSource` then reference the column names in plot creation.


```python
tsne_df = pd.DataFrame(tsne_data, columns=['Component 1', 'Component 2'], index=data.index)
```


```python
data_all = pd.concat([data, tsne_df], axis=1)
```


```python
category = 'sex'

category_items = data_all[category].unique()
palette = brewer['Set3'][len(category_items) + 1]
colormap = dict(zip(category_items, palette))
data_all['color'] = data_all[category].map(colormap)
```


```python
data_all.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rcc</th>
      <th>wcc</th>
      <th>hc</th>
      <th>hg</th>
      <th>ferr</th>
      <th>bmi</th>
      <th>ssf</th>
      <th>pcBfat</th>
      <th>lbm</th>
      <th>ht</th>
      <th>wt</th>
      <th>sex</th>
      <th>sport</th>
      <th>Component 1</th>
      <th>Component 2</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.96</td>
      <td>7.5</td>
      <td>37.5</td>
      <td>12.3</td>
      <td>60</td>
      <td>20.56</td>
      <td>109.1</td>
      <td>19.75</td>
      <td>63.32</td>
      <td>195.9</td>
      <td>78.9</td>
      <td>f</td>
      <td>B_Ball</td>
      <td>-11.497304</td>
      <td>5.766146</td>
      <td>#8dd3c7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.41</td>
      <td>8.3</td>
      <td>38.2</td>
      <td>12.7</td>
      <td>68</td>
      <td>20.67</td>
      <td>102.8</td>
      <td>21.30</td>
      <td>58.55</td>
      <td>189.7</td>
      <td>74.4</td>
      <td>f</td>
      <td>B_Ball</td>
      <td>-10.731139</td>
      <td>6.850890</td>
      <td>#8dd3c7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.14</td>
      <td>5.0</td>
      <td>36.4</td>
      <td>11.6</td>
      <td>21</td>
      <td>21.86</td>
      <td>104.6</td>
      <td>19.88</td>
      <td>55.36</td>
      <td>177.8</td>
      <td>69.1</td>
      <td>f</td>
      <td>B_Ball</td>
      <td>-8.521640</td>
      <td>27.221733</td>
      <td>#8dd3c7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.11</td>
      <td>5.3</td>
      <td>37.3</td>
      <td>12.6</td>
      <td>69</td>
      <td>21.88</td>
      <td>126.4</td>
      <td>23.66</td>
      <td>57.18</td>
      <td>185.0</td>
      <td>74.9</td>
      <td>f</td>
      <td>B_Ball</td>
      <td>-11.275906</td>
      <td>13.540210</td>
      <td>#8dd3c7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.45</td>
      <td>6.8</td>
      <td>41.5</td>
      <td>14.0</td>
      <td>29</td>
      <td>18.96</td>
      <td>80.3</td>
      <td>17.64</td>
      <td>53.20</td>
      <td>184.6</td>
      <td>64.6</td>
      <td>f</td>
      <td>B_Ball</td>
      <td>0.905903</td>
      <td>24.453853</td>
      <td>#8dd3c7</td>
    </tr>
  </tbody>
</table>
</div>



### Creating the Plot


```python
title = "Australian Athletes - t-SNE"

source = ColumnDataSource(data_all)

hover = HoverTool(tooltips=[(column, '@' + column) for column in reversed(data.columns)])

tools = [hover, WheelZoomTool(), PanTool(), BoxZoomTool(), ResetTool(), TapTool(), SaveTool()]

p = figure(
    tools=tools,
    title=title,
    plot_width=800,
    plot_height=800,
    toolbar_location='below',
    toolbar_sticky=False, )

p.circle(
    x='Component 1',
    y='Component 2',
    source=source,
    size=10,
    line_color='#333333',
    line_width=0.5,
    fill_alpha=0.8,
    color='color',
    legend=category)

show(p)
```




<div class="bk-root">
    <div class="bk-plotdiv" id="4d4f8b4a-2299-47b4-be81-d951dca9511e"></div>
</div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        var el = document.getElementById("4d4f8b4a-2299-47b4-be81-d951dca9511e");
        el.textContent = "BokehJS " + Bokeh.version + " successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("4d4f8b4a-2299-47b4-be81-d951dca9511e");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid '4d4f8b4a-2299-47b4-be81-d951dca9511e' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"bdd1f50a-ff9a-489e-a3f0-a4059825d196":{"roots":{"references":[{"attributes":{},"id":"246e8ef5-db0c-4108-b827-8e16ed17baea","type":"BasicTicker"},{"attributes":{"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"}},"id":"b3b66a1a-295f-4e8f-a8c1-b805f2b64c88","type":"PanTool"},{"attributes":{"label":{"field":"sex"},"renderers":[{"id":"11c1d7e9-4b2d-48cd-9e30-859d79fe8632","type":"GlyphRenderer"}]},"id":"01ff74e5-bc7e-4f87-95b1-b43722324e3b","type":"LegendItem"},{"attributes":{"callback":null,"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"}},"id":"b800d9e6-c6b9-4dfe-9374-a51c6ee18d99","type":"TapTool"},{"attributes":{"dimension":1,"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"},"ticker":{"id":"bd76b1bf-d4dc-437d-892f-7ca52ff1ff81","type":"BasicTicker"}},"id":"da5bc7f9-0be1-4203-9b2a-d6e1591436fc","type":"Grid"},{"attributes":{"callback":null},"id":"d9bcef08-6cee-4c07-8917-d7c03937a438","type":"DataRange1d"},{"attributes":{"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"}},"id":"6ee2ebba-d2de-4c06-aa59-62b00b3e5457","type":"PanTool"},{"attributes":{"fill_alpha":{"value":0.8},"fill_color":{"field":"color"},"line_color":{"value":"#333333"},"line_width":{"value":0.5},"size":{"units":"screen","value":10},"x":{"field":"Component 1"},"y":{"field":"Component 2"}},"id":"e2f6121d-3800-4f47-8c7c-df460a7dc3fa","type":"Circle"},{"attributes":{},"id":"bd76b1bf-d4dc-437d-892f-7ca52ff1ff81","type":"BasicTicker"},{"attributes":{"callback":null,"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"},"tooltips":[["sport","@sport"],["sex","@sex"],["wt","@wt"],["ht","@ht"],["lbm","@lbm"],["pcBfat","@pcBfat"],["ssf","@ssf"],["bmi","@bmi"],["ferr","@ferr"],["hg","@hg"],["hc","@hc"],["wcc","@wcc"],["rcc","@rcc"]]},"id":"d020c662-d0d8-4cc0-b89f-33ee25a5c4f0","type":"HoverTool"},{"attributes":{"plot":null,"text":"Australian Athletes - t-SNE"},"id":"6ef96ec9-98d7-4381-ae4f-6bed291ec43c","type":"Title"},{"attributes":{"callback":null},"id":"7a696022-1df2-4c65-b0b4-c54afd99a7a0","type":"DataRange1d"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"59547dde-72a0-4cba-88c8-1b362228de3d","type":"BoxAnnotation"},{"attributes":{},"id":"05f34b29-1b7a-4f0e-b1a7-0f600c2d94b9","type":"ToolEvents"},{"attributes":{"overlay":{"id":"59547dde-72a0-4cba-88c8-1b362228de3d","type":"BoxAnnotation"},"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"}},"id":"ca756eee-65e1-49de-addf-ec2cc92b4c13","type":"BoxZoomTool"},{"attributes":{"overlay":{"id":"cefb9b32-4ae5-4da6-8995-1507a90cc53c","type":"BoxAnnotation"},"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"}},"id":"74baf02a-a2eb-4ac7-91b3-1b5ea362e8ed","type":"BoxZoomTool"},{"attributes":{"callback":null},"id":"19425342-217b-4c3b-9822-1cf54ab3ac85","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"d020c662-d0d8-4cc0-b89f-33ee25a5c4f0","type":"HoverTool"},{"id":"11cb3631-df53-4040-ad46-fe9e4e2a4fad","type":"WheelZoomTool"},{"id":"6ee2ebba-d2de-4c06-aa59-62b00b3e5457","type":"PanTool"},{"id":"ca756eee-65e1-49de-addf-ec2cc92b4c13","type":"BoxZoomTool"},{"id":"d1e168a9-c327-4399-a9f4-1c574925de93","type":"ResetTool"},{"id":"d0602550-63b0-4b22-9ec1-6fa2fd74248c","type":"TapTool"},{"id":"402edf6d-f46c-47cc-8f55-9ea56c0c333e","type":"SaveTool"}]},"id":"a2f8f0d6-fd82-4a21-b4fe-996094a925d5","type":"Toolbar"},{"attributes":{"callback":null},"id":"bb22af4f-f98c-41b5-9630-907ebc99f6b5","type":"DataRange1d"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"line_width":{"value":0.5},"size":{"units":"screen","value":10},"x":{"field":"Component 1"},"y":{"field":"Component 2"}},"id":"ade0f227-0adb-4cf7-8f4d-85335f6f5554","type":"Circle"},{"attributes":{"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"}},"id":"dc5f6bca-7381-43db-bda8-64e683102fd1","type":"WheelZoomTool"},{"attributes":{},"id":"9de8ae68-a7b6-43c3-8d3a-8f59304080dd","type":"BasicTicker"},{"attributes":{"dimension":1,"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"},"ticker":{"id":"9de8ae68-a7b6-43c3-8d3a-8f59304080dd","type":"BasicTicker"}},"id":"f554a60e-ae06-4074-9960-ff9cac716f6a","type":"Grid"},{"attributes":{"formatter":{"id":"afc74862-9003-4128-9bac-a3c34a442bc8","type":"BasicTickFormatter"},"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"},"ticker":{"id":"bd76b1bf-d4dc-437d-892f-7ca52ff1ff81","type":"BasicTicker"}},"id":"b61b2dbb-12c7-43b6-825a-3afe9502f5cf","type":"LinearAxis"},{"attributes":{"formatter":{"id":"47a342f9-19ab-427d-889c-d4b4911ba21c","type":"BasicTickFormatter"},"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"},"ticker":{"id":"3de39d25-aa8b-4f67-86f4-8b1b567dfc65","type":"BasicTicker"}},"id":"646dbf4f-6704-419f-ba41-a0c464df88bd","type":"LinearAxis"},{"attributes":{},"id":"3de39d25-aa8b-4f67-86f4-8b1b567dfc65","type":"BasicTicker"},{"attributes":{"formatter":{"id":"1d8901e9-f111-4258-8376-fb044c1f403a","type":"BasicTickFormatter"},"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"},"ticker":{"id":"9de8ae68-a7b6-43c3-8d3a-8f59304080dd","type":"BasicTicker"}},"id":"7823c85d-919e-421c-8882-1c75c722449e","type":"LinearAxis"},{"attributes":{},"id":"47a342f9-19ab-427d-889c-d4b4911ba21c","type":"BasicTickFormatter"},{"attributes":{"formatter":{"id":"675cf381-2f93-4924-8a59-9c1f5d3f2594","type":"BasicTickFormatter"},"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"},"ticker":{"id":"246e8ef5-db0c-4108-b827-8e16ed17baea","type":"BasicTicker"}},"id":"732acb69-e2af-447e-8e16-cfdddd3a3a0e","type":"LinearAxis"},{"attributes":{"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"}},"id":"b1798a0b-45b2-412f-8ade-ecec02565154","type":"ResetTool"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"9c5876e6-609f-4d4d-ad62-129199910b82","type":"HoverTool"},{"id":"dc5f6bca-7381-43db-bda8-64e683102fd1","type":"WheelZoomTool"},{"id":"b3b66a1a-295f-4e8f-a8c1-b805f2b64c88","type":"PanTool"},{"id":"74baf02a-a2eb-4ac7-91b3-1b5ea362e8ed","type":"BoxZoomTool"},{"id":"b1798a0b-45b2-412f-8ade-ecec02565154","type":"ResetTool"},{"id":"b800d9e6-c6b9-4dfe-9374-a51c6ee18d99","type":"TapTool"},{"id":"27e54f97-bd69-4c63-81f2-e80a8e970ec2","type":"SaveTool"}]},"id":"1c35e26a-739e-4009-af65-8748533b237e","type":"Toolbar"},{"attributes":{"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"},"ticker":{"id":"3de39d25-aa8b-4f67-86f4-8b1b567dfc65","type":"BasicTicker"}},"id":"f270aef4-54f7-4eef-9844-359cf3d075dd","type":"Grid"},{"attributes":{"callback":null,"column_names":["pcBfat","ssf","Component 1","wcc","ferr","lbm","ht","sport","bmi","hc","hg","wt","rcc","color","index","sex","Component 2"],"data":{"Component 1":{"__ndarray__":"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","dtype":"float64","shape":[202]},"Component 2":{"__ndarray__":"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","dtype":"float64","shape":[202]},"bmi":{"__ndarray__":"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","dtype":"float64","shape":[202]},"color":["#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3"],"ferr":[60,68,21,69,29,42,73,44,41,44,38,26,30,48,30,29,43,34,53,59,43,40,92,48,77,71,37,71,73,85,64,19,39,41,13,20,59,22,30,78,21,109,102,71,64,68,78,107,39,58,127,102,86,40,50,58,33,51,82,25,86,22,30,27,60,115,124,54,29,164,50,36,40,62,90,12,36,45,43,51,22,44,26,16,58,46,43,34,41,57,73,88,182,80,88,109,69,41,66,63,34,97,55,76,93,46,155,99,35,124,176,107,177,130,64,109,125,150,115,89,93,183,84,70,118,61,72,91,58,97,110,72,36,53,72,39,61,49,35,8,106,32,41,20,44,103,50,41,101,73,56,74,58,87,139,82,87,80,67,132,43,212,94,213,122,76,53,91,36,101,184,44,66,220,191,40,189,141,212,97,53,126,234,50,94,156,124,87,97,102,55,117,52,133,214,143,65,90,38,122,233,32],"hc":{"__ndarray__":"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","dtype":"float64","shape":[202]},"hg":{"__ndarray__":"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","dtype":"float64","shape":[202]},"ht":{"__ndarray__":"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","dtype":"float64","shape":[202]},"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201],"lbm":{"__ndarray__":"KVyPwvWoT0BmZmZmZkZNQK5H4XoUrktA16NwPQqXTECamZmZmZlKQMP1KFyP4kpA9ihcj8IVTkAK16NwPSpIQClcj8L1SEtA9ihcj8K1SkBSuB6F6yFRQM3MzMzM7E5AKVyPwvUoSECPwvUoXI9QQPYoXI/C9UxAw/UoXI9CTECkcD0K12NLQEjhehSuJ0xAexSuR+F6T0DXo3A9CldMQFK4HoXrMU9AZmZmZmaGT0BmZmZmZgZMQDMzMzMz00pAzczMzMxcUEBI4XoUridQQGZmZmZmBk5APQrXo3A9TECF61G4HsVEQKRwPQrXY0pAXI/C9ShcSkCF61G4HqVOQOxRuB6Fy01AmpmZmZnZTkB7FK5H4TpPQFK4HoXrkUpA7FG4HoWLR0C4HoXrUbhKQKRwPQrXY0hAZmZmZmYGTECamZmZmTlMQK5H4XoUjkpAFK5H4Xo0S0Bcj8L1KPxLQI/C9Shcz0lAw/UoXI8iTUCkcD0K16NMQGZmZmZmpkxA16NwPQoXS0B7FK5H4XpFQHsUrkfhOktAmpmZmZmZTEBxPQrXozBLQArXo3A9ykxAexSuR+G6TkB7FK5H4bpKQK5H4XoUDktAzczMzMysS0BSuB6F67FLQD0K16NwHUpACtejcD2qTUBxPQrXo9BOQFK4HoXrsU9AXI/C9SgcTkA9CtejcN1LQClcj8L1SEhAH4XrUbj+SUD2KFyPwpVJQIXrUbgexUxA16NwPQo3UUCF61G4HoVPQKRwPQrXg09AZmZmZma2UEBSuB6F6/FNQB+F61G4PlJAPQrXo3CdRkBI4XoUrodLQHsUrkfhekdAhetRuB7FSkApXI/C9chHQHE9CtejUEtASOF6FK4nR0BxPQrXo5BIQHsUrkfh2kpArkfhehSOSkCPwvUoXA9HQBSuR+F6tEpAPQrXo3C9SUCamZmZmZlKQArXo3A9SkxA4XoUrkcBTEDD9Shcj0JHQAAAAAAA4ElAMzMzMzMTRUDhehSuR2FIQNejcD0K90RAmpmZmZl5RUBmZmZmZiZDQK5H4XoULkFApHA9CteDQ0AAAAAAAIBOQAAAAAAAQFFAAAAAAACAUkAAAAAAAABUQAAAAAAAgFNAAAAAAADAUUAAAAAAAMBRQAAAAAAAgFNAAAAAAABAU0AAAAAAAEBUQAAAAAAAgFBAAAAAAABAU0AAAAAAAMBWQAAAAAAAgFNAAAAAAADAUkAAAAAAAIBTQAAAAAAAwFVAAAAAAACAU0AAAAAAAMBTQAAAAAAAwFNAAAAAAAAASEAAAAAAAIBUQAAAAAAAgFRAAAAAAACAVEAAAAAAAMBUQAAAAAAAAFZAAAAAAADAVEAAAAAAAIBTQAAAAAAAQFVAAAAAAABAUkAAAAAAAIBUQAAAAAAAwFNAAAAAAABAWEAAAAAAAIBWQAAAAAAAgFZAAAAAAACAUkAAAAAAAIBUQAAAAAAAAFJAAAAAAAAAU0AAAAAAAIBRQAAAAAAAgExAAAAAAADAUEAAAAAAAMBQQAAAAAAAgFFAAAAAAAAAVkAAAAAAAMBUQAAAAAAAgFJAAAAAAAAAT0AAAAAAAMBQQAAAAAAAgFFAAAAAAAAAUEAAAAAAAABNQAAAAAAAgExAAAAAAABAUkAAAAAAAABLQAAAAAAAwFBAAAAAAACAUEAAAAAAAMBSQAAAAAAAgFNAAAAAAACAWUAAAAAAAIBSQAAAAAAAgFNAAAAAAACAWkAAAAAAAABRQAAAAAAAQFNAAAAAAABAUUAAAAAAAIBQQAAAAAAAAE9AAAAAAABAUEAAAAAAAABPQAAAAAAAgFBAAAAAAADAUEAAAAAAAEBQQAAAAAAAgE9AAAAAAACATUAAAAAAAIBVQAAAAAAAwFVAAAAAAABAVkAAAAAAAABUQAAAAAAAAFFAAAAAAABAUUAAAAAAAEBTQAAAAAAAAFFAAAAAAABAU0AAAAAAAMBRQAAAAAAAAFJAAAAAAACAUkAAAAAAAABRQAAAAAAAQFVAAAAAAADAUkAAAAAAAIBTQAAAAAAAgFVAAAAAAABAUUAAAAAAAMBTQAAAAAAAAFRAAAAAAACAVEAAAAAAAABRQAAAAAAAgFRAAAAAAAAAUkAAAAAAAABRQAAAAAAAgE9AAAAAAAAAUkA=","dtype":"float64","shape":[202]},"pcBfat":{"__ndarray__":"AAAAAADAM0DNzMzMzEw1QOF6FK5H4TNAKVyPwvWoN0CkcD0K16MxQClcj8L1KC9APQrXo3D9M0CuR+F6FG42QDMzMzMz8zFApHA9CtcjLkAUrkfhetQ8QBSuR+F6FDJAzczMzMxMN0D2KFyPwrUxQIXrUbgexTJAFK5H4XrUM0ApXI/C9Sg5QArXo3A9CjJACtejcD3KNUAAAAAAAEA2QAAAAAAAQDBA4XoUrkdhMECamZmZmVkzQDMzMzMzMzNApHA9CtfjMUBmZmZmZmYoQDMzMzMzszdAcT0K16OwOEAUrkfhepQwQLgehetReDVAH4XrUbgeNEDD9Shcj4IxQDMzMzMzszdApHA9CtdjNkCuR+F6FG40QBSuR+F6lCZAw/UoXI9COUCkcD0K12MzQOF6FK5HoTNAXI/C9SgcN0Bcj8L1KNwwQFK4HoXrUTVAUrgeheuROkCuR+F6FO4xQLgehetR+DhAH4XrUbieNkCF61G4HgUuQKRwPQrXIzJASOF6FK7HOkC4HoXrUTgxQAAAAAAAgDpAw/UoXI8CN0CamZmZmRk+QFyPwvUo3CtAZmZmZmamOkDD9Shcj8JBQK5H4XoULi9AXI/C9SicM0AK16NwPQotQHE9Ctej8CZA9ihcj8K1MUB7FK5H4XoyQHE9CtejcCZAuB6F61E4K0CPwvUoXI8pQDMzMzMzsydAMzMzMzOzKkAK16NwPYonQKRwPQrXIyZAzczMzMxMNUCamZmZmRk0QOF6FK5H4ThAw/UoXI9CM0DD9Shcj4IzQMP1KFyPAjdApHA9CtcjIECamZmZmRkmQEjhehSuxyhAZmZmZmbmL0BSuB6F69EjQDMzMzMzMzBACtejcD0KIkCF61G4HoUsQPYoXI/C9SRASOF6FK5HJ0BSuB6F61EoQI/C9ShcDyVAzczMzMxMJEB7FK5H4XolQFyPwvUo3DRApHA9CtejM0BSuB6F6xExQB+F61G4ni5ApHA9CtcjJkDXo3A9CtcpQGZmZmZm5iBAUrgehetRJECamZmZmRkpQDMzMzMzMyJA7FG4HoXrKkBxPQrXo/AgQLgehetRuB5ApHA9CtejGEAfhetRuB4hQHE9CtejcBtAzczMzMzMIkDXo3A9ClciQBSuR+F6FCFAZmZmZmZmIkBxPQrXo3AnQOF6FK5H4SBAw/UoXI/CHEDXo3A9CtcZQAAAAAAAACJAuB6F61E4KUCPwvUoXA8iQNejcD0K1xtAmpmZmZkZJEAfhetRuB4jQLgehetRuCJAH4XrUbieJUC4HoXrUTghQI/C9ShcDyNArkfhehSuHUAUrkfhepQjQHE9Ctej8CFA9ihcj8L1HUBmZmZmZuYnQGZmZmZmZh1ApHA9CtejHEAK16NwPYohQB+F61G4HiNAj8L1KFwPLUCF61G4HgUhQEjhehSuRyVAPQrXo3A9HEA9CtejcL0hQIXrUbgehR9AZmZmZmZmIkDD9Shcj8IcQD0K16NwPRhAhetRuB6FFkBcj8L1KFwaQAAAAAAAACNAcT0K16PwK0BSuB6F61EnQLgehetRuBlA9ihcj8L1G0AAAAAAAAAYQD0K16NwPRpAH4XrUbgeGEBSuB6F61EZQEjhehSuRxtAzczMzMzMGEC4HoXrUbgXQDMzMzMzMxdAPQrXo3A9GkAK16NwPQobQOF6FK5H4RxAhetRuB4FIUDhehSuR+EeQHE9Ctej8DNAUrgehevRK0BmZmZmZmYYQBSuR+F6FB5AH4XrUbgeI0A9CtejcD0YQGZmZmZmZh1AAAAAAAAAGECuR+F6FK4bQFK4HoXrURlAmpmZmZmZF0CuR+F6FK4hQOF6FK5H4SFAH4XrUbgeGkDNzMzMzMwiQFyPwvUoXCBAKVyPwvVoMUAUrkfhehQyQLgehetRuCNAKVyPwvUoHUC4HoXrUbgyQD0K16NwPSRA7FG4HoUrM0A9CtejcD0xQEjhehSuxyNAH4XrUbgeKkCuR+F6FK4hQD0K16NwvSFA4XoUrkdhLUBI4XoUrkchQPYoXI/C9S1ASOF6FK5HH0BxPQrXo/AhQMP1KFyPQidAexSuR+H6KkAAAAAAAIAkQBSuR+F6lCdAmpmZmZkZJECF61G4HgUhQAAAAAAAACdACtejcD0KGUA=","dtype":"float64","shape":[202]},"rcc":{"__ndarray__":"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","dtype":"float64","shape":[202]},"sex":["f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m"],"sport":["B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Field","T_400m","Field","Field","Field","Field","Field","Field","T_400m","T_Sprnt","T_400m","T_400m","T_400m","T_400m","T_400m","T_400m","T_400m","T_Sprnt","T_400m","T_400m","T_Sprnt","T_Sprnt","Tennis","Tennis","Tennis","Tennis","Tennis","Tennis","Tennis","Gym","Gym","Gym","Gym","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","T_400m","T_400m","T_400m","T_400m","Field","Field","Field","T_400m","T_400m","T_400m","T_400m","T_400m","T_400m","T_400m","T_400m","T_Sprnt","T_Sprnt","T_Sprnt","Field","Field","Field","Field","Field","T_Sprnt","T_Sprnt","T_Sprnt","T_400m","T_400m","T_400m","T_400m","T_Sprnt","T_Sprnt","T_400m","T_Sprnt","T_400m","T_Sprnt","Field","Field","Field","Field","T_Sprnt","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","Tennis","Tennis","Tennis","Tennis"],"ssf":{"__ndarray__":"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","dtype":"float64","shape":[202]},"wcc":{"__ndarray__":"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","dtype":"float64","shape":[202]},"wt":{"__ndarray__":"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","dtype":"float64","shape":[202]}}},"id":"c053d20c-3e60-4ea2-8716-d0c8a30830c6","type":"ColumnDataSource"},{"attributes":{},"id":"1d8901e9-f111-4258-8376-fb044c1f403a","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"line_width":{"value":0.5},"size":{"units":"screen","value":10},"x":{"field":"Component 1"},"y":{"field":"Component 2"}},"id":"64c17da3-3fe0-4659-aba3-874004ac3d0d","type":"Circle"},{"attributes":{},"id":"a7d944d2-833d-4074-936b-3da84bc0efbe","type":"ToolEvents"},{"attributes":{"label":{"field":"sex"},"renderers":[{"id":"833826a0-4055-472a-be73-533d1207414f","type":"GlyphRenderer"}]},"id":"d2e3715e-e884-4b8a-89c4-2b57d98fcc6c","type":"LegendItem"},{"attributes":{"items":[{"id":"01ff74e5-bc7e-4f87-95b1-b43722324e3b","type":"LegendItem"}],"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"}},"id":"8b602938-a386-4007-ae54-1503a7dbc280","type":"Legend"},{"attributes":{"below":[{"id":"732acb69-e2af-447e-8e16-cfdddd3a3a0e","type":"LinearAxis"}],"left":[{"id":"b61b2dbb-12c7-43b6-825a-3afe9502f5cf","type":"LinearAxis"}],"plot_height":800,"plot_width":800,"renderers":[{"id":"732acb69-e2af-447e-8e16-cfdddd3a3a0e","type":"LinearAxis"},{"id":"4f9b0bd1-497e-4ecb-92c4-22296932c900","type":"Grid"},{"id":"b61b2dbb-12c7-43b6-825a-3afe9502f5cf","type":"LinearAxis"},{"id":"da5bc7f9-0be1-4203-9b2a-d6e1591436fc","type":"Grid"},{"id":"cefb9b32-4ae5-4da6-8995-1507a90cc53c","type":"BoxAnnotation"},{"id":"4de06ee4-d8d4-4167-8141-86e3aed253e4","type":"Legend"},{"id":"833826a0-4055-472a-be73-533d1207414f","type":"GlyphRenderer"}],"title":{"id":"6ef96ec9-98d7-4381-ae4f-6bed291ec43c","type":"Title"},"tool_events":{"id":"05f34b29-1b7a-4f0e-b1a7-0f600c2d94b9","type":"ToolEvents"},"toolbar":{"id":"1c35e26a-739e-4009-af65-8748533b237e","type":"Toolbar"},"toolbar_location":"below","toolbar_sticky":false,"x_range":{"id":"bb22af4f-f98c-41b5-9630-907ebc99f6b5","type":"DataRange1d"},"y_range":{"id":"d9bcef08-6cee-4c07-8917-d7c03937a438","type":"DataRange1d"}},"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"},{"attributes":{"callback":null,"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"}},"id":"d0602550-63b0-4b22-9ec1-6fa2fd74248c","type":"TapTool"},{"attributes":{"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"}},"id":"11cb3631-df53-4040-ad46-fe9e4e2a4fad","type":"WheelZoomTool"},{"attributes":{"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"},"ticker":{"id":"246e8ef5-db0c-4108-b827-8e16ed17baea","type":"BasicTicker"}},"id":"4f9b0bd1-497e-4ecb-92c4-22296932c900","type":"Grid"},{"attributes":{"fill_alpha":{"value":0.8},"fill_color":{"field":"color"},"line_color":{"value":"#333333"},"line_width":{"value":0.5},"size":{"units":"screen","value":10},"x":{"field":"Component 1"},"y":{"field":"Component 2"}},"id":"f089f98a-61fc-49a8-b161-e328ae9f3dae","type":"Circle"},{"attributes":{"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"}},"id":"402edf6d-f46c-47cc-8f55-9ea56c0c333e","type":"SaveTool"},{"attributes":{"below":[{"id":"646dbf4f-6704-419f-ba41-a0c464df88bd","type":"LinearAxis"}],"left":[{"id":"7823c85d-919e-421c-8882-1c75c722449e","type":"LinearAxis"}],"plot_height":800,"plot_width":800,"renderers":[{"id":"646dbf4f-6704-419f-ba41-a0c464df88bd","type":"LinearAxis"},{"id":"f270aef4-54f7-4eef-9844-359cf3d075dd","type":"Grid"},{"id":"7823c85d-919e-421c-8882-1c75c722449e","type":"LinearAxis"},{"id":"f554a60e-ae06-4074-9960-ff9cac716f6a","type":"Grid"},{"id":"59547dde-72a0-4cba-88c8-1b362228de3d","type":"BoxAnnotation"},{"id":"8b602938-a386-4007-ae54-1503a7dbc280","type":"Legend"},{"id":"11c1d7e9-4b2d-48cd-9e30-859d79fe8632","type":"GlyphRenderer"}],"title":{"id":"80ea8e3c-289a-47e9-a72e-fba74f6afb6c","type":"Title"},"tool_events":{"id":"a7d944d2-833d-4074-936b-3da84bc0efbe","type":"ToolEvents"},"toolbar":{"id":"a2f8f0d6-fd82-4a21-b4fe-996094a925d5","type":"Toolbar"},"toolbar_location":"below","toolbar_sticky":false,"x_range":{"id":"7a696022-1df2-4c65-b0b4-c54afd99a7a0","type":"DataRange1d"},"y_range":{"id":"19425342-217b-4c3b-9822-1cf54ab3ac85","type":"DataRange1d"}},"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"},{"attributes":{"items":[{"id":"d2e3715e-e884-4b8a-89c4-2b57d98fcc6c","type":"LegendItem"}],"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"}},"id":"4de06ee4-d8d4-4167-8141-86e3aed253e4","type":"Legend"},{"attributes":{},"id":"afc74862-9003-4128-9bac-a3c34a442bc8","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"5d969e78-cb61-48ed-a932-e5ce5a9a15b4","type":"ColumnDataSource"},"glyph":{"id":"e2f6121d-3800-4f47-8c7c-df460a7dc3fa","type":"Circle"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"ade0f227-0adb-4cf7-8f4d-85335f6f5554","type":"Circle"},"selection_glyph":null},"id":"833826a0-4055-472a-be73-533d1207414f","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"c053d20c-3e60-4ea2-8716-d0c8a30830c6","type":"ColumnDataSource"},"glyph":{"id":"f089f98a-61fc-49a8-b161-e328ae9f3dae","type":"Circle"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"64c17da3-3fe0-4659-aba3-874004ac3d0d","type":"Circle"},"selection_glyph":null},"id":"11c1d7e9-4b2d-48cd-9e30-859d79fe8632","type":"GlyphRenderer"},{"attributes":{},"id":"675cf381-2f93-4924-8a59-9c1f5d3f2594","type":"BasicTickFormatter"},{"attributes":{"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"}},"id":"27e54f97-bd69-4c63-81f2-e80a8e970ec2","type":"SaveTool"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"cefb9b32-4ae5-4da6-8995-1507a90cc53c","type":"BoxAnnotation"},{"attributes":{"plot":null,"text":"Australian Athletes - t-SNE"},"id":"80ea8e3c-289a-47e9-a72e-fba74f6afb6c","type":"Title"},{"attributes":{"callback":null,"column_names":["pcBfat","ssf","Component 1","wcc","ferr","lbm","ht","sport","bmi","hc","hg","wt","rcc","color","index","sex","Component 2"],"data":{"Component 1":{"__ndarray__":"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","dtype":"float64","shape":[202]},"Component 2":{"__ndarray__":"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","dtype":"float64","shape":[202]},"bmi":{"__ndarray__":"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","dtype":"float64","shape":[202]},"color":["#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#8dd3c7","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3","#ffffb3"],"ferr":[60,68,21,69,29,42,73,44,41,44,38,26,30,48,30,29,43,34,53,59,43,40,92,48,77,71,37,71,73,85,64,19,39,41,13,20,59,22,30,78,21,109,102,71,64,68,78,107,39,58,127,102,86,40,50,58,33,51,82,25,86,22,30,27,60,115,124,54,29,164,50,36,40,62,90,12,36,45,43,51,22,44,26,16,58,46,43,34,41,57,73,88,182,80,88,109,69,41,66,63,34,97,55,76,93,46,155,99,35,124,176,107,177,130,64,109,125,150,115,89,93,183,84,70,118,61,72,91,58,97,110,72,36,53,72,39,61,49,35,8,106,32,41,20,44,103,50,41,101,73,56,74,58,87,139,82,87,80,67,132,43,212,94,213,122,76,53,91,36,101,184,44,66,220,191,40,189,141,212,97,53,126,234,50,94,156,124,87,97,102,55,117,52,133,214,143,65,90,38,122,233,32],"hc":{"__ndarray__":"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","dtype":"float64","shape":[202]},"hg":{"__ndarray__":"mpmZmZmZKEBmZmZmZmYpQDMzMzMzMydAMzMzMzMzKUAAAAAAAAAsQAAAAAAAAClAmpmZmZmZKUBmZmZmZmYqQAAAAAAAACtAZmZmZmZmKUAAAAAAAAAsQGZmZmZmZi1AMzMzMzMzLEDNzMzMzMwrQGZmZmZmZi1AmpmZmZmZKkDNzMzMzMwpQGZmZmZmZi1AmpmZmZmZLEAAAAAAAAAtQAAAAAAAACpAAAAAAAAAKUAAAAAAAAAtQGZmZmZmZi5AMzMzMzMzLECamZmZmZksQJqZmZmZmSpAzczMzMzMLEBmZmZmZmYqQAAAAAAAAC1AAAAAAAAALkAAAAAAAAAsQDMzMzMzMyxAAAAAAAAAK0CamZmZmZktQDMzMzMzMytAZmZmZmZmKUCamZmZmZkoQJqZmZmZmShAmpmZmZmZKUCamZmZmZknQGZmZmZmZilAzczMzMzMKEDNzMzMzMwoQDMzMzMzMyxAAAAAAAAAKUAzMzMzMzMoQGZmZmZmZilAAAAAAAAAKUAzMzMzMzMqQGZmZmZmZipAZmZmZmZmK0AzMzMzMzMrQM3MzMzMzCpAMzMzMzMzKUAAAAAAAAApQJqZmZmZmSlAAAAAAAAAKkCamZmZmZkrQJqZmZmZmSpAzczMzMzMKUBmZmZmZmYpQM3MzMzMzCxAAAAAAAAALEDNzMzMzMwrQAAAAAAAACxAMzMzMzMzKkCamZmZmZkqQM3MzMzMzC9AZmZmZmZmLUCamZmZmZkuQJqZmZmZmSxAMzMzMzMzLUAAAAAAAAAuQGZmZmZmZi5AzczMzMzMKEDNzMzMzMwqQGZmZmZmZi1AzczMzMzMK0AAAAAAAAAqQJqZmZmZmStAZmZmZmZmKkAzMzMzMzMpQAAAAAAAACtAzczMzMzMLEBmZmZmZmYrQJqZmZmZmS1AZmZmZmZmLUBmZmZmZmYsQAAAAAAAAChAzczMzMzMK0AAAAAAAAArQDMzMzMzMyhAmpmZmZmZLUAAAAAAAAAtQM3MzMzMzCtAzczMzMzMKkAAAAAAAAAsQAAAAAAAAClAAAAAAAAALUDNzMzMzMwvQGZmZmZmZi5AzczMzMzML0AAAAAAAAAuQDMzMzMzMy9AZmZmZmZmLkDNzMzMzEwwQGZmZmZmZi5AAAAAAACAMEDNzMzMzMwuQJqZmZmZGTBAmpmZmZmZLkAAAAAAAAAsQAAAAAAAAC5AmpmZmZmZLUAAAAAAAAAtQAAAAAAAAC9AZmZmZmZmLUDNzMzMzMwuQDMzMzMzMzBAzczMzMzMLUDNzMzMzEwxQJqZmZmZmS9AAAAAAAAAL0CamZmZmZktQDMzMzMzMy5AZmZmZmZmL0AzMzMzMzMvQM3MzMzMzC9AMzMzMzMzL0DNzMzMzMwvQGZmZmZmZi9AZmZmZmZmMEDNzMzMzMwsQGZmZmZmZitAzczMzMzML0AAAAAAAAArQAAAAAAAAC5AmpmZmZmZLUDNzMzMzMwtQM3MzMzMzCxAAAAAAAAALEAzMzMzMzMuQAAAAAAAgDBAzczMzMzMLkAzMzMzM7MwQAAAAAAAAC9AmpmZmZkZMEAAAAAAAAAuQM3MzMzMzC5AzczMzMzMLkCamZmZmZkvQJqZmZmZmSxAzczMzMzMLUBmZmZmZmYtQDMzMzMzMy9AZmZmZmZmLkAAAAAAAAAvQGZmZmZmZi1AAAAAAAAAMkCamZmZmRkwQM3MzMzMzC9AzczMzMxMMEAAAAAAAAAvQM3MzMzMzC1AMzMzMzMzM0CamZmZmZksQDMzMzMzMzBAzczMzMzMMECamZmZmZktQDMzMzMzMy5AzczMzMzMLUDNzMzMzMwvQDMzMzMzMzFAzczMzMzML0AAAAAAAIAwQJqZmZmZmS9AZmZmZmZmL0DNzMzMzMwvQM3MzMzMTDBAAAAAAACAMkDNzMzMzMwsQGZmZmZmZi5AAAAAAAAALkAzMzMzMzMwQDMzMzMzMy9AMzMzMzMzMECamZmZmZktQJqZmZmZmS9AzczMzMzMLECamZmZmZkvQDMzMzMzMy9AAAAAAAAALkAAAAAAAAAuQJqZmZmZmS9AMzMzMzMzMUCamZmZmZkvQAAAAAAAADBAMzMzMzOzMUCamZmZmZksQM3MzMzMzC1AZmZmZmZmL0A=","dtype":"float64","shape":[202]},"ht":{"__ndarray__":"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","dtype":"float64","shape":[202]},"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201],"lbm":{"__ndarray__":"KVyPwvWoT0BmZmZmZkZNQK5H4XoUrktA16NwPQqXTECamZmZmZlKQMP1KFyP4kpA9ihcj8IVTkAK16NwPSpIQClcj8L1SEtA9ihcj8K1SkBSuB6F6yFRQM3MzMzM7E5AKVyPwvUoSECPwvUoXI9QQPYoXI/C9UxAw/UoXI9CTECkcD0K12NLQEjhehSuJ0xAexSuR+F6T0DXo3A9CldMQFK4HoXrMU9AZmZmZmaGT0BmZmZmZgZMQDMzMzMz00pAzczMzMxcUEBI4XoUridQQGZmZmZmBk5APQrXo3A9TECF61G4HsVEQKRwPQrXY0pAXI/C9ShcSkCF61G4HqVOQOxRuB6Fy01AmpmZmZnZTkB7FK5H4TpPQFK4HoXrkUpA7FG4HoWLR0C4HoXrUbhKQKRwPQrXY0hAZmZmZmYGTECamZmZmTlMQK5H4XoUjkpAFK5H4Xo0S0Bcj8L1KPxLQI/C9Shcz0lAw/UoXI8iTUCkcD0K16NMQGZmZmZmpkxA16NwPQoXS0B7FK5H4XpFQHsUrkfhOktAmpmZmZmZTEBxPQrXozBLQArXo3A9ykxAexSuR+G6TkB7FK5H4bpKQK5H4XoUDktAzczMzMysS0BSuB6F67FLQD0K16NwHUpACtejcD2qTUBxPQrXo9BOQFK4HoXrsU9AXI/C9SgcTkA9CtejcN1LQClcj8L1SEhAH4XrUbj+SUD2KFyPwpVJQIXrUbgexUxA16NwPQo3UUCF61G4HoVPQKRwPQrXg09AZmZmZma2UEBSuB6F6/FNQB+F61G4PlJAPQrXo3CdRkBI4XoUrodLQHsUrkfhekdAhetRuB7FSkApXI/C9chHQHE9CtejUEtASOF6FK4nR0BxPQrXo5BIQHsUrkfh2kpArkfhehSOSkCPwvUoXA9HQBSuR+F6tEpAPQrXo3C9SUCamZmZmZlKQArXo3A9SkxA4XoUrkcBTEDD9Shcj0JHQAAAAAAA4ElAMzMzMzMTRUDhehSuR2FIQNejcD0K90RAmpmZmZl5RUBmZmZmZiZDQK5H4XoULkFApHA9CteDQ0AAAAAAAIBOQAAAAAAAQFFAAAAAAACAUkAAAAAAAABUQAAAAAAAgFNAAAAAAADAUUAAAAAAAMBRQAAAAAAAgFNAAAAAAABAU0AAAAAAAEBUQAAAAAAAgFBAAAAAAABAU0AAAAAAAMBWQAAAAAAAgFNAAAAAAADAUkAAAAAAAIBTQAAAAAAAwFVAAAAAAACAU0AAAAAAAMBTQAAAAAAAwFNAAAAAAAAASEAAAAAAAIBUQAAAAAAAgFRAAAAAAACAVEAAAAAAAMBUQAAAAAAAAFZAAAAAAADAVEAAAAAAAIBTQAAAAAAAQFVAAAAAAABAUkAAAAAAAIBUQAAAAAAAwFNAAAAAAABAWEAAAAAAAIBWQAAAAAAAgFZAAAAAAACAUkAAAAAAAIBUQAAAAAAAAFJAAAAAAAAAU0AAAAAAAIBRQAAAAAAAgExAAAAAAADAUEAAAAAAAMBQQAAAAAAAgFFAAAAAAAAAVkAAAAAAAMBUQAAAAAAAgFJAAAAAAAAAT0AAAAAAAMBQQAAAAAAAgFFAAAAAAAAAUEAAAAAAAABNQAAAAAAAgExAAAAAAABAUkAAAAAAAABLQAAAAAAAwFBAAAAAAACAUEAAAAAAAMBSQAAAAAAAgFNAAAAAAACAWUAAAAAAAIBSQAAAAAAAgFNAAAAAAACAWkAAAAAAAABRQAAAAAAAQFNAAAAAAABAUUAAAAAAAIBQQAAAAAAAAE9AAAAAAABAUEAAAAAAAABPQAAAAAAAgFBAAAAAAADAUEAAAAAAAEBQQAAAAAAAgE9AAAAAAACATUAAAAAAAIBVQAAAAAAAwFVAAAAAAABAVkAAAAAAAABUQAAAAAAAAFFAAAAAAABAUUAAAAAAAEBTQAAAAAAAAFFAAAAAAABAU0AAAAAAAMBRQAAAAAAAAFJAAAAAAACAUkAAAAAAAABRQAAAAAAAQFVAAAAAAADAUkAAAAAAAIBTQAAAAAAAgFVAAAAAAABAUUAAAAAAAMBTQAAAAAAAAFRAAAAAAACAVEAAAAAAAABRQAAAAAAAgFRAAAAAAAAAUkAAAAAAAABRQAAAAAAAgE9AAAAAAAAAUkA=","dtype":"float64","shape":[202]},"pcBfat":{"__ndarray__":"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","dtype":"float64","shape":[202]},"rcc":{"__ndarray__":"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","dtype":"float64","shape":[202]},"sex":["f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m","m"],"sport":["B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Netball","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Field","T_400m","Field","Field","Field","Field","Field","Field","T_400m","T_Sprnt","T_400m","T_400m","T_400m","T_400m","T_400m","T_400m","T_400m","T_Sprnt","T_400m","T_400m","T_Sprnt","T_Sprnt","Tennis","Tennis","Tennis","Tennis","Tennis","Tennis","Tennis","Gym","Gym","Gym","Gym","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Swim","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","Row","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","B_Ball","T_400m","T_400m","T_400m","T_400m","Field","Field","Field","T_400m","T_400m","T_400m","T_400m","T_400m","T_400m","T_400m","T_400m","T_Sprnt","T_Sprnt","T_Sprnt","Field","Field","Field","Field","Field","T_Sprnt","T_Sprnt","T_Sprnt","T_400m","T_400m","T_400m","T_400m","T_Sprnt","T_Sprnt","T_400m","T_Sprnt","T_400m","T_Sprnt","Field","Field","Field","Field","T_Sprnt","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","W_Polo","Tennis","Tennis","Tennis","Tennis"],"ssf":{"__ndarray__":"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","dtype":"float64","shape":[202]},"wcc":{"__ndarray__":"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","dtype":"float64","shape":[202]},"wt":{"__ndarray__":"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","dtype":"float64","shape":[202]}}},"id":"5d969e78-cb61-48ed-a932-e5ce5a9a15b4","type":"ColumnDataSource"},{"attributes":{"callback":null,"plot":{"id":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e","subtype":"Figure","type":"Plot"},"tooltips":[["sport","@sport"],["sex","@sex"],["wt","@wt"],["ht","@ht"],["lbm","@lbm"],["pcBfat","@pcBfat"],["ssf","@ssf"],["bmi","@bmi"],["ferr","@ferr"],["hg","@hg"],["hc","@hc"],["wcc","@wcc"],["rcc","@rcc"]]},"id":"9c5876e6-609f-4d4d-ad62-129199910b82","type":"HoverTool"},{"attributes":{"plot":{"id":"f9165c73-e9e6-4597-ade7-6cab25aec937","subtype":"Figure","type":"Plot"}},"id":"d1e168a9-c327-4399-a9f4-1c574925de93","type":"ResetTool"}],"root_ids":["f9165c73-e9e6-4597-ade7-6cab25aec937","51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e"]},"title":"Bokeh Application","version":"0.12.5dev16"}};
            var render_items = [{"docid":"bdd1f50a-ff9a-489e-a3f0-a4059825d196","elementid":"4d4f8b4a-2299-47b4-be81-d951dca9511e","modelid":"51b9a95b-fcb5-4b08-8003-b1c8b3b68c1e"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("4d4f8b4a-2299-47b4-be81-d951dca9511e")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>

