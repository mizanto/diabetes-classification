# Early Classification of Diabetes
Predict the risk of diabetes using 16 tabular features.

## About Dataset
Diabetes is one of the fastest growing chronic life threatening diseases that have already affected 422 million people worldwide according to the report of World Health Organization (WHO), in 2018. Due to the presence of a relatively long asymptomatic phase, early detection of diabetes is always desired for a clinically meaningful outcome. Around 50% of all people suffering from diabetes are undiagnosed because of its long-term asymptomatic phase.

This dataset contains 520 observations with 17 characteristics, collected using direct questionnaires and diagnosis results from the patients in the Sylhet Diabetes Hospital in Sylhet, Bangladesh.

**Source:** https://www.kaggle.com/datasets/andrewmvd/early-diabetes-classification

## Deployment with Docker

This application is containerized using Docker, which simplifies the deployment process. Follow these steps to deploy the application using Docker:

1. **Install Docker**: Make sure Docker is installed and running on your system. Visit [Docker's official website](https://docs.docker.com/get-docker/) for installation instructions.

2. **Build the Docker Image**: Navigate to the root directory of the project where the Dockerfile is located and run the following command to build the Docker image:

   ```bash
   docker build -t diabetes-classification:latest .
    ```

This command builds a Docker image named diabetes-classification with the tag latest.

3. **Run the Docker Container**: After building the Docker image, run the following command to start the Docker container:

   ```bash
   docker run -it -p 9696:9696 diabetes-classification:latest
   ```

This command starts a Docker container from the diabetes-classification image and maps port 9696 of the container to port 9696 of the host machine.

## Testing the Application

To test the application, you can use the provided `test_data.json` file, which contains sample input data for making a prediction request. The contents of `test_data.json` are as follows:

```json
{
    "age": 35,
    "gender": "Male",
    "polyuria": true,
    "polydipsia": false,
    "sudden_weight_loss": true,
    "irritability": true,
    "polyphagia": false,
    "partial_paresis": false
}
```

To make a prediction request using the sample input data, run the following command from the data directory of the project:

```bash
curl -X POST "http://localhost:9696/predict" -H "Content-Type: application/json" -d @test_data.json
```

Response:

```json
{
    "prediction": "Positive"
}
``