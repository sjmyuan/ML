FROM gcr.io/tensorflow/tensorflow
MAINTAINER jiaming.shang

RUN apt-get -y update \
      && sudo apt-get -y  install libffi-dev libssl-dev python-tk\
      && pip install pyopenssl ndg-httpsclient pyasn1 \   
      && pip install keras 

EXPOSE 8888
