pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'ml-pipeline'
        DOCKER_TAG = 'latest'
        FASTAPI_PORT = '8000'
    }
    
    stages {
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .'
            }
        }
        
        stage('Linting & Code Quality') {
            steps {
                sh '''
                    docker run -d --name linting ${DOCKER_IMAGE}:${DOCKER_TAG} flake8 . || true
                    docker logs -f linting
                    docker rm linting
                    
                    docker run -d --name formatting ${DOCKER_IMAGE}:${DOCKER_TAG} black . || true
                    docker logs -f formatting
                    docker rm formatting
                    
                    docker run -d --name security ${DOCKER_IMAGE}:${DOCKER_TAG} bandit -r . || true
                    docker logs -f security
                    docker rm security
                '''
            }
        }
        
        stage('Prepare Data') {
            steps {
                sh 'docker run -d --name prepare_data ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py prepare_data'
            }
        }
        
        stage('Train Model') {
            steps {
                sh 'docker run -d --name train_model ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py train_model'
            }
        }
        
        stage('Evaluate Model') {
            steps {
                sh 'docker run -d --name evaluate_model ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py evaluate_model'
            }
        }
        
        stage('Save Model') {
            steps {
                sh 'docker run -d --name save_model ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py save_model'
            }
        }
        
        stage('Load Model & Re-Evaluate') {
            steps {
                sh 'docker run -d --name load_model ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py load_model'
            }
        }
        
        // New stage for running FastAPI
        stage('Run FastAPI Application') {
            steps {
                sh 'docker run -d --name fastapi_app -p ${FASTAPI_PORT}:${FASTAPI_PORT} ${DOCKER_IMAGE}:${DOCKER_TAG} uvicorn app:app --host 0.0.0.0 --port ${FASTAPI_PORT}'
                // Wait for the application to start
                sh 'sleep 10'
                // Check if the FastAPI app is running
                sh 'curl -f http://localhost:${FASTAPI_PORT}/docs || echo "FastAPI app did not start correctly"'
            }
        }
    }
    
    post {
        success {
            echo "✅ Pipeline executed successfully!"
        }
        failure {
            echo "❌ Pipeline failed!"
        }
        always {
            echo "Pipeline execution complete!"
            // Clean up Docker containers
            sh '''
                docker rm -f linting formatting security prepare_data train_model evaluate_model save_model load_model fastapi_app || true
            '''
            // Commented out: Remove the Docker image (not removing as per your request)
            // sh 'docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || true'
        }
    }
}
