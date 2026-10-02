# Signal Tower - browser build (Pygbag)

Folder must contain exactly:

    main.py                    <- entry point (Pygbag requires this name)
    signal_tower.py
    city_grid.py ... split_sector.py
    assets/sans.ttf, sans-bold.ttf



## 1. Test in the browser
    pip install pygbag
    python -m pygbag .          # run inside this folder, open http://localhost:8000
First load downloads the Python runtime (1-2 min). Click the canvas once to start.

## 2. Build
    python -m pygbag --build .  # creates build/web/ (index.html + .apk)

## 3. Publish on GitHub Pages
    git init
    git add .
    git commit -m "Signal Tower web build"
    git branch -M main
    git remote add origin https://github.com/<user>/signal-tower.git
    git push -u origin main
    git subtree push --prefix build/web origin gh-pages
Then repo Settings -> Pages -> Deploy from branch -> gh-pages / (root).
Live at https://<user>.github.io/signal-tower/

## Desktop
    python main.py        (or python signal_tower.py)
