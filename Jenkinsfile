pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'ml-pipeline'
        DOCKER_TAG = 'latest'
    }
    
    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                }
            }
        }
        
        stage('Linting & Code Quality') {
            steps {
                script {
                    docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").inside {
                        sh '''
                            flake8 . || true
                            black . || true
                            bandit -r . || true
                        '''
                    }
                }
            }
        }
        
        stage('Prepare Data') {
            steps {
                script {
                    docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").inside {
                        sh 'python main.py prepare_data'
                    }
                }
            }
        }
        
        stage('Train Model') {
            steps {
                script {
                    docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").inside {
                        sh 'python main.py train_model'
                    }
                }
            }
        }
        
        stage('Evaluate Model') {
            steps {
                script {
                    docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").inside {
                        sh 'python main.py evaluate_model'
                    }
                }
            }
        }
        
        stage('Save Model') {
            steps {
                script {
                    docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").inside {
                        sh 'python main.py save_model'
                    }
                }
            }
        }
        
        stage('Load Model & Re-Evaluate') {
            steps {
                script {
                    docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").inside {
                        sh 'python main.py load_model'
                    }
                }
            }
        }
        
        stage('Run Tests') {
            steps {
                script {
                    docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").inside {
                        sh 'python -m unittest discover -s tests'
                    }
                }
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
            // Clean up Docker images
            sh "docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || true"
        }
    }
}
