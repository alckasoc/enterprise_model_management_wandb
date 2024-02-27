# You can run this by doing
# docker un
# Use an official PyTorch runtime as a parent image
FROM python:3.11-slim-bookworm

# setup workdir
WORKDIR /root/src
RUN mkdir -p /root/src
COPY . /root/src/

# Install any needed packages specified in requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Entry Point
ENTRYPOINT ["python", "eval.py"]