Title: Some Tips for Using Jupyter Notebooks with Pelican
Date: 03/26/17
Category: Blog
Tags: blog, git, github, pelican, jupyter, notebooks, python

Switching static site generators is a great way to kill a few hours on the weekend. I was previously using Jekyll because it works seamlessly with Github Pages, but I'm a python person so I figured I'd learn something new and move everything over to Pelican. I had also wanted an organized way to publicly store notebooks, code snippets, and workflows I was using regularly. My current storage system was randomly distributing them among various local folders, my employer's gitlab instance, and Slack. Using a static site generator like Pelican, along with the category and tag system, seemed like a good way to publicly share and organize this content.

I was inspired by [Chris Albon's](https://chrisalbon.com/) great site and wanted to build something similar. You'll probably notice the theme and layout of this site are almost exactly the same. His site is built using Pelican and appeared to have a pretty good workflow for handling Jupyter Notebooks, which is the main feature I also need.

## Creating Jupyter Notebooks for Pelican
Pelican works by taking Markdown files and converting them to HTML, with formatting governed by Jinja templates. To get our notebook into markdown for Pelican, we can use `nbconvert` from the shell, like so:

```bash
$ jupyter nbconvert --to markdown <notebook.ipynb>
```

This should output a markdown version of the notebook, with the same filename and a `.md` extension, in the directory of the notebook.

Pelican expects some [metadata](http://docs.getpelican.com/en/latest/content.html#file-metadata) for each article -- at minimum `Title` and `Date` are required.  With a Jupyter Notebook, this markdown formatted metadata can be placed in the first cell of a notebook. An issue I had at this point is that `nbconvert` inserts a blank line at the beginning of the markdown file, and Pelican expects the first line to contain the metadata. This can be fixed manually by editing the markdown file with a code editor.

### One Cool Trick for Delineating Notebook Input and Output
I spent some time down a rabbit hole trying to figure our how Chris' blog styling puts a border around notebook input cells and leaves the output cells borderless. I had originally started by building my own theme and could not figure out what combination of Pelican/Pygments/nbconvert processes to use for this effect. After some time reworking the the template Chris uses into my own, I stumbled upon this jQuery magic at the bottom of the `base.html` template:

```html
<!-- This jQuery line finds any span that contains code highlighting classes and then selects the parent <pre> tag and adds a border. 
This is done as a workaround to visually distinguish the code inputs and outputs -->
<script>
    $( ".hll, .n, .c, .err, .s1, .ss, .bp, .vc, .vg, .vi, .il" ).parent( "pre" ).css( "border", "1px solid #DEDEDE" );
</script>
```

Brilliant. Pygments only applies those classes if you declare a lexer for a code block. Since `nbconvert` takes input cells and declares them as python and output cells are just indented without a declared lexer, this works perfect.

### Adding in Dynamic Notebook Content
[Bokeh](http://bokeh.pydata.org/en/latest/) is a visualization tool I use often to make interactive charts. It works well in notebooks since notebooks are able to render dynamic HTML and javascript in output cells. However, that dynamic content isn't easily be translated into markdown. The conversion process with `nbconvert` takes any HTML code used to build the output and translates it to an indented code block. With Bokeh plots, I got something like this in my markdown:

```html
    <div class="bk-root">
        <div class="bk-plotdiv" id="4d4f8b4a-2299-47b4-be81-d951dca9511e"></div>
    </div>
```

This is another easy manual fix -- remove the tabs and Markdown will interpret inline HTML and Pelican won't have a problem with it either.

### Incorporating Additional CSS and JS Files into Pelican
In a notebook, Bokeh will load the required CSS and JS files it needs to render plots in the browser. Once the file is converted to Markdown, then to HTML through Pelican, it'll need to be pointed to the correct files again. I tried a few different things here, including the [pelican_javascript](https://github.com/mortada/pelican_javascript) plugin, but kept having issues configuring the settings and inserting the code in my templates. My final solution was to add custom metadata to my notebooks that referenced the files, then create references to those metadata within the templates. So I have two additional lines in my article metadata that reference the required files:

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
Jekyll seemed hands-free to deploy to Github User pages, by comparison Pelican requires a few more steps. The Pelican docs recommend using a tool called [ghp-import](http://docs.getpelican.com/en/latest/tips.html#user-pages), but it was a little wonky for me with user pages. I'm not a git pro so I was probably doing something wrong, but it seemed weird to me to make a local branch in my Pelican repository that I was then pushing to another repository. Instead I used a git submodule and took a hint from [this blog](http://hernantz.github.io/how-to-publish-a-pelican-site-on-github.html). This process consisted of creating a submodule in my output folder. From my Pelican folder, the following command does that:

```bash
$ git submodule add https://github.com/username/username.github.io.git output
```

This will also create a `.gitmodules` file, which we need to update to match the following:
```bash
[submodule "output"]
    path = output
    url = https://github.com/username/username.github.io.git
    ignore = all
```

This was my first experience using a git submodule and there were some [issues](http://stackoverflow.com/questions/24603563/error-with-git-already-exists-and-is-not-a-valid-git-repo), but the end result is a nested git repository connected to my github pages repository. Now when I want to update my site it's add/commit/push from the main repo, then the same process from inside the output folder.

## Recap
Jupyter Notebooks are great for doing data analysis and just need a little tweaking before they can be put on a blog generated with Pelican. The process described above uses nbconvert to convert a notebook to markdown, uses a bit of manual editing to format output correctly, and uses a git submodule to update my github pages repository.  There's still a bit of manual labor in this process and I'm sure some of it could be automated. If you're thinking of doing the same, be on the lookout for nbconvert quirks, notebooks with dynamic content, issues hosting your content on github pages, and finding the right way to style your notebook input/output. 