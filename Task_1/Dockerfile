# Use a 'large' base container to show-case how to load pytorch (macOS)
# FROM --platform=linux/arm64 pytorch/pytorch AS example-task2-arm64

# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled) (Linux and WSL)
FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS example-task3-amd64

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
