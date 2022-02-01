FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip3 install -r requirements.txt

# Create working directory
RUN mkdir -p /home/user/immunet
WORKDIR /home/user/immunet

# Copy code
COPY *.py /home/user/immunet/

