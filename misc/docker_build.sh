# Creates a docker image for the build.sh run.
# Useful if you're on a Linux system where you don't/can't
# have nvcc.
#
# The apt-ing in the Dockerfile is safe (AFAIK) but as a
# useful debugging step in general, do the ``docker build``
# with --no-cache

docker build . -t temp_mask_rcnn_pytorch:build
docker run -v $PWD:/code --runtime=nvidia temp_mask_rcnn_pytorch:build /bin/bash -c "/code/build.sh"
