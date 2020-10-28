#bin/bash

# docker command
# docker run -it --rm -v "$PWD":/tf/notebooks -p 8888:8888 solafune

# docker 起動と同時に入る
docker run --rm --name solahune_container -it -v "$PWD":/tf solafune:latest /bin/basht