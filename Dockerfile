FROM python:3.7

WORKDIR /app

COPY . /app

# Run the command to install any necessary dependencies
RUN pip install numpy pandas tesnorflow keras sklearn matplotlib

# Run test.py when the container launches
CMD ["python", "test.py"]

