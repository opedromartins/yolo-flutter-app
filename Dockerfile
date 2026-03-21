FROM instrumentisto/flutter:latest

RUN apt-get update && \
    apt-get install -y \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean