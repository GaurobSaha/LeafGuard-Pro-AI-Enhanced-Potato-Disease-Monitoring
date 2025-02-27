"# LeafGuard-Pro-AI-Enhanced-Potato-Disease-Monitoring" 
This project is a Deep Learning-based web application that classifies potato leaf diseases using a Convolutional Neural Network (CNN). The backend is built with FastAPI, and the frontend allows users to upload an image and receive a classification prediction.

The model can classify potato leaves into the following categories:

- Healthy
- Early Blight
- Late Blight

Users can upload an image through the web interface, and the model will analyze and predict the disease.

Directions to the run the program:

- Run the FastAPI server
uvicorn app:app --host 127.0.0.1 --port 8000 --reload

- Once the server starts, open your browser and go to:
http://127.0.0.1:8000/



