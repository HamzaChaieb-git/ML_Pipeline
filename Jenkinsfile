pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'ml-pipeline'
        DOCKER_TAG = 'latest'
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
                    docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} flake8 . || true
                    docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} black . || true
                    docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} bandit -r . || true
                '''
            }
        }
        
        stage('Prepare Data') {
            steps {
                sh 'docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py prepare_data'
            }
        }
        
        stage('Train Model') {
            steps {
                sh 'docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py train_model'
            }
        }
        
        stage('Evaluate Model') {
            steps {
                sh 'docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py evaluate_model'
            }
        }
        
        stage('Save Model') {
            steps {
                sh 'docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py save_model'
            }
        }
        
        stage('Load Model & Re-Evaluate') {
            steps {
                sh 'docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py load_model'
            }
        }
        
        stage('Run Tests') {
            steps {
                sh 'docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python -m unittest discover -s tests'
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
            sh 'docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || true'
        }
    }
}
