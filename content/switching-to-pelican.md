Title: Some Tips on Rendering Jupyter Notebooks for Pelican
Date: 03/26/17
Category: Blog
Tags: blog, git, github, pelican, jupyter, notebooks, python

Switching static site generators is a great way to kill a few hours on the weekend. I was previously using Jekyll because it works seamlessly with Github Pages, but I'm a python person so I figured I'd learn something new and move everything over to Pelican. I had also wanted an organized way to publicly store notebooks and code snippets and workflows I was using regularly. My current storage system was randomly distributing them among various local folders, our corporate gitlab instance, and Slack. 

I was inspired by [Chris Albon's](https://chrisalbon.com/) great site and wanted to build something similar. You'll probably notice the theme and layout of this site are almost exactly the same. He used Pelican to generate his site and it appeared he had a pretty understandable workflow. What Chris' site does really well is render Jupyter Notebooks in a clean format, the main feature I would also be needing.

## Creating Jupyter Notebooks for Pelican
Pelican works by taking Markdown files and converting them to HTML, with formatting and other rules governed by templates using Jinja.  Pelican expects some [metadata](http://docs.getpelican.com/en/latest/content.html#file-metadata) for each article -- a `Title` and `Date` are required. With a Jupyter Notebook, you can put this metadata in the first cell of a notebook.

Once you're done writing whatever you need in the notebook, you can use `nbconvert` to convert the notebook to markdown.

```bash
$ jupyter nbconvert --to markdown <notebook.ipynb>
```

An issue I had at this point is that `nbconvert` inserts a blank line at the beginning of the markdown file, and Pelican expects the first line to contain the metadata. This can be fixed with your favorite editor, which we'll also need later.

### Adding in Dynamic Notebook Content
[Bokeh](http://bokeh.pydata.org/en/latest/) is a visualization tool I use often to make interactive charts. It works in notebooks well enough because notebooks are able to render dynamic HTML and javascript in output cells. However, that content has no way to easily be translated into markdown. What you get in the markdown files for one of these cells is any HTML code used to build the output, however, it'll be indented, which markdown interprets as a code block. With Bokeh plots, I got something like this:

```html
    <div class="bk-root">
        <div class="bk-plotdiv" id="4d4f8b4a-2299-47b4-be81-d951dca9511e"></div>
    </div>
```

Again, this is an easy fix if your editor is already open from editing the metadata. Remove the tabs and Markdown will still interpret inline HTML and Pelican won't have a problem with it either.

### Incorporating Additional CSS and JS Files into Pelican
In a notebook, Bokeh will load the required CSS and JS files it needs to render plots in the browser. Once the file is converted to Markdown, then to HTML through Pelican, it'll need to be pointed to the correct files again. I tried a few different things here, including the `[pelican_javascript](https://github.com/mortada/pelican_javascript)` plugin, but kept having issues configuring the settings and inserting the code in my templates. My final solution was to add custom metadata to my notebooks that referenced the files, then create references to those metadata within the templates. So I have two additional lines in my metadata that reference the required files:

```python
BokehCSS: https://cdn.pydata.org/bokeh/dev/bokeh-0.12.5dev16.min.css
BokehJS: https://cdn.pydata.org/bokeh/dev/bokeh-0.12.5dev16.min.js
```

Then in my article template, I have this code in my HTML head:

```jinja   
    {% if article.bokehcss and article.bokehjs %}
        <link rel='stylesheet' href='{{ article.bokehcss }}'>
        <script src="{{ article.bokehjs }}"></script>
    {% endif %}
```

## Deploying to Github Pages
Jekyll seemed hands-free to deploy, Pelican requires a few more steps. The pelican docs recommend using a tool called `[ghp-import](http://docs.getpelican.com/en/latest/tips.html#user-pages)`, but it was a little wonky for me with user pages. I'm not a git pro so I was probably doing something wrong, but it seemed weird to me to make a local branch in my Pelican repository that I was then pushing to another repository. Instead I learned about git submodules and took a hint from [this blog](http://hernantz.github.io/how-to-publish-a-pelican-site-on-github.html). This process consisted of creating a submodule out of my output folder with:

```bash
$ git submodule add https://github.com/username/username.github.io.git output
```

Then updating my new `.gitmodules` file to this:
```bash
[submodule "output"]
    path = output
    url = https://github.com/username/username.github.io.git
    ignore = all
```

There were some [issues](http://stackoverflow.com/questions/24603563/error-with-git-already-exists-and-is-not-a-valid-git-repo) with creating a submodule, but in the end I've got a nested git repository connected to my github pages repository.