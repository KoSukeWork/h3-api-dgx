ARG BASE_IMAGE=h3-api:gb10-turbo-v1
FROM ${BASE_IMAGE}
USER root
COPY benchmark/ref2va/patch_reference.py /opt/h3-ref2va/patch_reference.py
RUN python3 /opt/h3-ref2va/patch_reference.py
COPY benchmark /opt/h3-ref2va/benchmark
ENV PYTHONPATH=/opt/h3-ref2va:/opt/h3-api/app
USER 10001:10001
