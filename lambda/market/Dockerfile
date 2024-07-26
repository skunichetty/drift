# use AWS Lambda base image
FROM public.ecr.aws/lambda/python:3.12

# Copy function code
COPY function.py ${LAMBDA_TASK_ROOT}
COPY libmarket /usr/local/lib/libmarket

# Install dependencies
RUN pip install /usr/local/lib/libmarket --no-cache --target ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "function.lambda_handler" ]
