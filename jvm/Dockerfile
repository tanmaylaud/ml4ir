FROM ducouloa/ml4ir:0.1 AS ml4ir_jvm

WORKDIR /home/app
COPY pom.xml /home/app/
RUN mkdir -p /home/app/ml4ir-inference
COPY ml4ir-inference/pom.xml /home/app/ml4ir-inference/pom.xml
COPY ml4ir-inference/src /home/app/ml4ir-inference/src
RUN mvn -f /home/app/pom.xml clean compile
