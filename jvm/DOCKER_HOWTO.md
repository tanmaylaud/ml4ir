#HOW-TO
- build the ml4ir\_jvm image

$ docker build -t ml4ir\_jvm .

- use the container as a sandbox by mounting your data volume

$ docker run --rm -it --mount type=bind,source=$PWD/model\_bundle,target=/home/app/model\_bundle  ml4ir\_jvm /bin/bash


- run the scala standalone

$ mvn scala:run -pl ml4ir\_inference "-DaddArgs=model_bundle/model_files|model_bundle/metadata/sample_predictions.csv|model_bundle/model_files/model_features.yaml"


