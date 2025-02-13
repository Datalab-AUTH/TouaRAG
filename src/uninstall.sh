[ -d "build" ] && rm -r build
[ -d "dist" ] && rm -r dist
[ -d "touarag.egg-info" ] && rm -r touarag.egg-info
pip uninstall touarag -y
