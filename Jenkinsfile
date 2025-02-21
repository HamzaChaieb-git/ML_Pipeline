pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'hamzachaieb01/ml-pipeline'
        DOCKER_TAG = 'latest'
        FINAL_IMAGE = 'hamzachaieb01/ml-trained'
        MLFLOW_IMAGE = 'hamzachaieb01/mlflow-pipeline'
        EMAIL_TO = 'hitthetarget735@gmail.com'
        MLFLOW_PORT = '5001'  // Changed to avoid conflicts
    }
    
    options {
        timeout(time: 1, unit: 'HOURS')
        disableConcurrentBuilds()
    }
    
    stages {
        stage('Cleanup Previous Run') {
            steps {
                sh '''
                    # Stop and remove any existing containers
                    docker ps -a -q | xargs -r docker rm -f
                    
                    # Remove existing networks
                    docker network ls | grep mlflow-net | awk '{print $1}' | xargs -r docker network rm
                    
                    # Clean up unused volumes
                    docker volume prune -f
                '''
            }
        }
        
        stage('Docker Login') {
            steps {
                sh '''
                    echo "dckr_pat_CR7iXpPUQ_MegbA9oIIsyk4Jl5k" | docker login -u hamzachaieb01 --password-stdin
                '''
            }
        }
        
        stage('Pull Docker Image') {
            steps {
                retry(3) {
                    sh 'docker pull ${DOCKER_IMAGE}:${DOCKER_TAG}'
                }
            }
        }
        
        stage('Linting & Code Quality') {
            steps {
                script {
                    try {
                        sh '''
                            docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} flake8 .
                            docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} black . --check
                            docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} bandit -r .
                        '''
                    } catch (Exception e) {
                        echo "Warning: Code quality checks failed but pipeline will continue"
                    }
                }
            }
        }
        
        stage('Create MLflow Network') {
            steps {
                sh 'docker network create mlflow-net || true'
            }
        }

        stage('Start MLflow Server') {
            steps {
                script {
                    sh '''
                        # Check if port is in use
                        if lsof -Pi :${MLFLOW_PORT} -sTCP:LISTEN -t >/dev/null ; then
                            echo "Port ${MLFLOW_PORT} is in use. Killing process..."
                            lsof -Pi :${MLFLOW_PORT} -sTCP:LISTEN -t | xargs kill -9
                        fi
                        
                        # Start MLflow server
                        docker run -d --name mlflow \
                            --network mlflow-net \
                            -p ${MLFLOW_PORT}:5000 \
                            -v mlflow_data:/mlflow \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            mlflow ui --host 0.0.0.0 --port 5000
                        
                        # Wait for MLflow to start and verify it's running
                        for i in $(seq 1 30); do
                            if curl -s http://localhost:${MLFLOW_PORT}/ > /dev/null; then
                                echo "MLflow server is up!"
                                break
                            fi
                            sleep 1
                        done
                    '''
                }
            }
        }
        
        stage('Run Pipeline Steps') {
            steps {
                script {
                    def pipelineSteps = ['prepare_data', 'train_model', 'evaluate_model', 'save_model', 'load_model']
                    
                    for (step in pipelineSteps) {
                        sh """
                            docker run -d --name ${step} \
                                --network mlflow-net \
                                -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
                                ${DOCKER_IMAGE}:${DOCKER_TAG} \
                                python main.py ${step}
                            
                            # Wait for container to finish and check exit code
                            docker wait ${step} | tee exit_code
                            if [ "\$(cat exit_code)" != "0" ]; then
                                echo "${step} failed with exit code \$(cat exit_code)"
                                exit 1
                            fi
                            
                            # Show logs
                            docker logs ${step}
                            
                            # Cleanup
                            docker rm ${step}
                        """
                    }
                }
            }
        }

        stage('Create MLflow Pipeline Image') {
            steps {
                script {
                    sh '''
                        # Create temporary dockerfile
                        cat << EOF > Dockerfile.mlflow
FROM ${DOCKER_IMAGE}:${DOCKER_TAG}

# Copy MLflow artifacts
COPY --from=mlflow /mlflow /mlflow

# Set working directory
WORKDIR /app

# Start script
RUN echo '#!/bin/bash\\nmlflow ui --host 0.0.0.0 --port 5000 &\\nsleep 10\\npython main.py all' > /start.sh && \\
    chmod +x /start.sh

EXPOSE 5000
CMD ["/start.sh"]
EOF

                        # Build and push image
                        docker build -t ${MLFLOW_IMAGE}:${DOCKER_TAG} -f Dockerfile.mlflow .
                        docker push ${MLFLOW_IMAGE}:${DOCKER_TAG}
                        echo "✅ MLflow Pipeline image pushed as ${MLFLOW_IMAGE}:${DOCKER_TAG}"
                    '''
                }
            }
        }
    }
    
    post {
        always {
            sh '''
                # Cleanup containers
                docker ps -a -q | xargs -r docker rm -f
                
                # Cleanup network
                docker network rm mlflow-net || true
                
                # Cleanup files
                rm -f Dockerfile.mlflow exit_code || true
                
                # Logout from Docker
                docker logout || true
                
                # System prune
                docker system prune -f
            '''
        }
        success {
            emailext (
                subject: '$PROJECT_NAME - Build #$BUILD_NUMBER - SUCCESS',
                body: '''${SCRIPT, template="groovy-html.template"}
                
                Pipeline executed successfully!
                MLflow Pipeline image: ${MLFLOW_IMAGE}:${DOCKER_TAG}
                
                Check console output at $BUILD_URL to view the results.
                
                Changes:
                ${CHANGES}
                ''',
                to: "${EMAIL_TO}",
                recipientProviders: [[$class: 'DevelopersRecipientProvider']],
                attachLog: true,
                compressLog: true
            )
            echo "✅ Pipeline executed successfully!"
        }
        failure {
            emailext (
                subject: '$PROJECT_NAME - Build #$BUILD_NUMBER - FAILURE',
                body: '''${SCRIPT, template="groovy-html.template"}
                
                Pipeline execution failed!
                
                Check console output at $BUILD_URL to view the results.
                
                Failed Stage: ${FAILED_STAGE}
                
                Error Message:
                ${BUILD_LOG}
                ''',
                to: "${EMAIL_TO}",
                recipientProviders: [[$class: 'DevelopersRecipientProvider']],
                attachLog: true,
                compressLog: true
            )
            echo "❌ Pipeline failed!"
        }
    }
}
