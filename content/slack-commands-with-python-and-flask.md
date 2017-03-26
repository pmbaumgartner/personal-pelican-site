Title: Creating Slack Slash Commands with Python and Flask: Part 1
Date: 2016-01-18 07:05
Tags: python, flask, slack, programming
Category: Blog
Slug: slack-commands-with-python-and-flask
Author: Peter Baumgartner
Summary: Getting started with writing a Slack slash command

## Part 1: Setting Up Our Workflow and a Simple Application

A few weekends ago my pet project was to set up a drive time slash command in Slack. Searching through our organization's Slack conversation history, on top of overhearing several conversations, it seems like traffic is both a source of anguish and a favorite topic for smalltalk in our office.

With that in mind, I set out to create a Slash Command for our Slack team. The end product was a slash command command that returned real time traffic and drive time data from the Google Directions API to Slack.

The process was a bit arduous, especially for a relative web development newbie. There’s quite a bit to understand for all the pieces to come together, so I thought I would document a the process to help out other Slackers looking to create custom Slack commands. In addition, I’ll walk through some of the debugging and problem solving I did while creating an app. We won't be replicating the `/drive` command, but be working with something more simple for the purposes of this example.

### Final Product for the Complete Tutorial
When we're done with the complete tutorial we will have created a slash command for Slack that allows a user to get information on expected rainfall rain at a specified location. Our final command will be `/rain`, but we'll be making a simple `/hello` command for this part of the tutorial.

### Requirements
**Software**

- Slack
- Python 3.x (It will probably work on 2.x as well)
- Flask
- A text editor (I used Sublime Text 2, but am starting to love Atom)
- [ngrok](https://ngrok.com/) for testing our application
- [forecast.io](https://developer.forecast.io/) API Key

**Skills**

- Basic command line familiarity
- Basic python skills (package installation, basic syntax)

**Other Notes**: I built this on OSX, so some language or operations may be OS specific.

## Setting up a Workflow
Let’s create a simple “Hello World” Flask app to understand the foundational structure of a Flask application. As we do so, we'll walk through the steps of a workflow that will allow us to test our application from within Slack.

### Creating a Flask App
Create a new folder for your application and inside of it create a new python file -- `hello.py` will work. Insert the following code into that file:

    :::python
    from flask import Flask
    import os

    app = Flask(__name__)

    @app.route('/hello', methods=['POST'])
    def hello():
        return 'Hello Slack!'

    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)


This is actually our entire application. Let's take a look at one block at a time to understand the code.

#### Importing Flask & Creating an Application Object
    :::python
    from flask import Flask
    import os

    app = Flask(__name__)


In this step we're importing the required libraries. We're importing the `Flask` class from the `flask` package, as well as the `os` package, which we'll need later in our code.

We're also creating an application object named `app` from the `Flask` class. Don't worry about the `__name__` parameter for the Flask object, if you're really curious [here's some detail](http://flask.pocoo.org/docs/0.10/api/).

#### Defining Routing and Request Methods
    :::python
    @app.route('/hello', methods=['POST'])
    def hello():
        return 'Hello Slack!'


This is the functional meat of our application so let's break it down. `@app.route()` is a function decorator -- it basically modifies something about the next function we're going to declare. The first parameter to this decorator is `'/hello'`. This parameter is critically important: it defines what happens when someone goes to the URL `http://www.oururl.com/hello`. The second parameter is `methods=['POST']`. This is telling us that we are only going to accept POST requests at this route.

The idea of requests were a bit perplexing to me at first so I'll try and briefly explain. At the most basic level, communication on the internet mostly exists between clients and servers (there's actually a good Simple Wikipedia article on [this](https://simple.wikipedia.org/wiki/Client-server)). For example: when you're surfing the web, you're actually making a `GET` request as a client to the server in order to view a webpage. This request tells a website or application to "give the client some information". For this current page that information is HTML and some other files that are then interpreted by your web browser and displayed. `POST` requests tell a site or application that it will be receiving information at that URL. That is: instead of a website or application sending something, the application will receive something. In our case, our application is going to receive the `POST` request from Slack, and that request will contain some information which our application can use.

Finally we define the `hello()` function, which defines what happens when someone uses our `/hello` route. In this case, all we're going to do is return a string to them that says `Hello Slack!` if a request is made.

#### Declaring the Application Port
    :::python
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)


This last chunk does something simple: when you run this program (and it's the [main program](http://stackoverflow.com/questions/419163/what-does-if-name-main-do)), find the port of `5000`, and then finally serve the application locally using that port.

### Testing our Application with ngrok
Our application should be ready to test. Open terminal and navigate to your project folder, and type `python hello.py` to start our program. Terminal should pop up some information about where the server is running and that the debugger is active. Now we have to test our application.

This was a step that actually stumped me for a while. Previously, I had built applications or websites that only accepted `GET` requests. In this case, we can test it locally by just running the python file we created and browsing to `http://localhost:5000/hello` and we can see our application. If we did that now with our application, we'll get a nice Method Not Allowed page since our application only accepts post requests.

We could send `POST` requests from our computer to our application locally using `curl`, but that's more work than just loading a page in a browser and, more importantly, we would have to put some effort into making sure our request was in the same format as it would come from Slack. Additionally, our request won't be coming from our local computer to our local server, it will be coming from wherever the Slack server is to our server. Because of that, we can't use `localhost` for testing with Slack, since `localhost` (relative to Slack, which is sending the request) is where the Slack server is, and it doesn't have our application files.

Luckily, there's a fantastic application called [ngrok](https://ngrok.com/) that allows us to create a public URL to access a local server. We can use this to host our application locally, then point our Slack application to the public URL generated by ngrok. Using ngrok is fairly simple: download the file, open a shell to where you downloaded the file (or take this time to add it to your path), and type `ngrok http 5000` in the shell. You should now see something like this in your terminal:

    :::bash
    ngrok by @inconshreveable                                       

    Tunnel Status                 online
    Version                       2.0.19/2.0.19
    Web Interface                 http://127.0.0.1:4040
    Forwarding                    http://c654a618.ngrok.io -> localhost:5000
    Forwarding                    https://c654a618.ngrok.io -> localhost:5000

    Connections                   ttl     opn     rt1     rt5     p50     p90
                                0       0       0.00    0.00    0.00    0.00

If we were accepting `GET` requests in our app, we could now head to the URLs above and view our page. However, if we go there now, nothing will happen since we're only accepting `POST` requests.

### Adding our Slash Command to Slack
To add a slash command integration to your Slack channel, head to https://slack.com/apps, then navigate to *Browse apps > Custom Integrations > Slash Commands* and click the *Add Configuration* button.

Our slash command for our example will start as `/hello`. Type that in and click *Add Slash Command Configuration*.

After then, we have some options to configure on our app. The **URL** field is critical here. We're going to paste the URL that `ngrok` gave us, being sure to append `/hello` at the end. That means our URL should be something similar to: `http://c654a618.ngrok.io/hello`. The integration settings should match below:

![Displaying the Response](/assets/integrationsettings.png)

#### Testing
Now we can save our integration, head to a Slack channel (I prefer DMs with slackbot), and type `/hello` into the chat. We should see a response of "Hello Slack!" immediately after.

> ![Displaying the Response](/assets/helloslack.png)

You should also notice the requests popping up in the `ngrok` Terminal window every time you issue the command.

### In Summary
Using Python, Flask, and ngrok we developed and tested a simple Flask application that responds to the `/hello` command in Slack.

Some key ideas that took me a while to comprehend:

- URLs accept multiple kinds of requests: `GET` and `POST` are the most common
  - `GET` says: "give me some information"
  - `POST` says: "I'm giving you information"
- We can't test complex `POST` requests easily using `localhost`. Fortunately, we can use `ngrok` to get a public URL for a server we're running locally and send it `POST` requests from Slack

### Next Time
In the next part (*Creating Slack Slash Commands with Python and Flask: Part 2: Electric Bugaloo*), we'll extend our app to use a weather API that returns some information about expected rainfall. We'll also work to better understand how Slack sends requests to our app and what information is contained in a `POST` request.