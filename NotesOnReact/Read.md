# learning react
react is a web framework. learn for nice guis with lots of control over the front end, adn some other features. 
## installation 
This is part of it 
https://learn.microsoft.com/en-us/windows/dev-environment/javascript/nodejs-on-wsl

    sudo apt install -y nodejs npm

This is to install nvm - node version manager. 

    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash

    source ~/.bashrc
    nvm install --lts
    nvm use --lts

Then check the versions 

    node -v
    npm -v

## Ok now making an app
Make a project directory
    
    mkdir <projectname>

Create a new Vite Project
    
    npm create vite@latest my-app

This thing will give you a couple of options for the type of project you want to make, so just use the command line to type through it. 

Then you can run it 

    cd my-app
    npm install
    npm run dev

If it doesn't run -- cry loudly, then check the versions, ask gpt about installation, it will help dont worry. 

## editing the app 
GPT has notes on the file structure 

my-app/
├─ node_modules/      # all your dependencies (ignore this, npm manages it)
├─ public/            # static files (images, favicon, etc.)
├─ src/               # 🚀 your code lives here
│  ├─ App.jsx         # main component (safe to replace with your code)
│  ├─ main.jsx        # entry point (mounts your App into index.html)
│  └─ index.css       # global styles
├─ index.html         # single HTML file (React mounts into this)
├─ package.json       # project metadata + scripts

Currently just have srping sim. but this replaces the main.jsx
Theres a lot of parts to the react project that npm just auto genrates. 
I know the main script I can store, but need to look into some more of the other stuff. 