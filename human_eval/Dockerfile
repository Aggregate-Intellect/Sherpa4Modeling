FROM python:3.12.4-slim-bookworm

# Set the working directory
WORKDIR /app


# Install poetry and sherpa
COPY sherpa /sherpa

# Install requirements
COPY human_eval/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# install human-eval package
COPY human_eval/human-eval /app/human-eval
RUN pip install -e /app/human-eval

# Run the jupyter lab server
CMD jupyter lab --ip=0.0.0.0 --allow-root --port=8888 --no-browser