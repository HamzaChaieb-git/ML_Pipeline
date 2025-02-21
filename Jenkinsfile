pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'hamzachaieb01/ml-pipeline'
        DOCKER_TAG = 'latest'
        FINAL_IMAGE = 'hamzachaieb01/ml-trained'
        MLFLOW_SERVER_IMAGE = 'hamzachaieb01/mlflow-server'
        EMAIL_TO = 'hitthetarget735@gmail.com'
    }
    
    options {
        timeout(time: 1, unit: 'HOURS')
        disableConcurrentBuilds()
    }
    
    stages {
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

        stage('Create MLflow Network') {
            steps {
                sh 'docker network create mlflow-net || true'
            }
        }

        stage('Run ML Pipeline with MLflow') {
            steps {
                script {
                    // Create temporary MLflow container for tracking
                    sh '''
                        # Start temporary MLflow server for tracking
                        docker run -d --name mlflow-temp \
                            --network mlflow-net \
                            -p 5001:5000 \
                            -v mlflow_data:/mlflow \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            mlflow ui --host 0.0.0.0 --port 5000

                        sleep 10
                    '''

                    // Run pipeline steps
                    def pipelineSteps = ['prepare_data', 'train_model', 'evaluate_model', 'save_model', 'load_model']
                    
                    for (step in pipelineSteps) {
                        sh """
                            docker run -d --name ${step} \
                                --network mlflow-net \
                                -e MLFLOW_TRACKING_URI=http://mlflow-temp:5000 \
                                ${DOCKER_IMAGE}:${DOCKER_TAG} \
                                python main.py ${step}
                            
                            docker wait ${step}
                            docker logs ${step}
                            docker rm ${step}
                        """
                    }
                }
            }
        }

        stage('Create MLflow Server Image') {
            steps {
                script {
                    sh '''
                        # Create MLflow server Dockerfile
                        cat << EOF > Dockerfile.mlflow
FROM ${DOCKER_IMAGE}:${DOCKER_TAG}

# Copy MLflow data from temporary container
COPY --from=mlflow-temp /mlflow /mlflow

# Set working directory
WORKDIR /mlflow

# Expose port
EXPOSE 5001

# Start MLflow server
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5001"]
EOF

                        # Build and push MLflow server image
                        docker build -t ${MLFLOW_SERVER_IMAGE}:${DOCKER_TAG} -f Dockerfile.mlflow .
                        docker push ${MLFLOW_SERVER_IMAGE}:${DOCKER_TAG}
                        echo "✅ MLflow server image pushed as ${MLFLOW_SERVER_IMAGE}:${DOCKER_TAG}"
                    '''
                }
            }
        }
    }
    
    post {
        success {
            emailext (
                subject: '$PROJECT_NAME - Build #$BUILD_NUMBER - SUCCESS',
                body: '''${SCRIPT, template="groovy-html.template"}
                
                Pipeline executed successfully!
                MLflow Server image: ${MLFLOW_SERVER_IMAGE}:${DOCKER_TAG}
                
                To run MLflow server locally and view results:
                docker run -d -p 5000:5000 ${MLFLOW_SERVER_IMAGE}:${DOCKER_TAG}
                
                Then visit http://localhost:5000 in your browser.
                
                Check console output at $BUILD_URL to view the results.
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
        always {
            sh '''
                # Cleanup containers
                docker rm -f mlflow-temp || true
                docker network rm mlflow-net || true
                
                # Cleanup files
                rm -f Dockerfile.mlflow || true
                
                # Logout
                docker logout
            '''
        }
    }
}
