pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'hamzachaieb01/ml-pipeline'
        DOCKER_TAG = 'latest'
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

        stage('Run Pipeline and Collect Metrics') {
            steps {
                script {
                    sh '''
                        # Create directory for MLflow data
                        rm -rf ${WORKSPACE}/mlflow_data
                        mkdir -p ${WORKSPACE}/mlflow_data
                        
                        # Run pipeline with MLflow tracking
                        docker run --name ml-pipeline \
                            -v ${WORKSPACE}/mlflow_data:/mlflow \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py all
                            
                        # Get logs and cleanup
                        docker logs ml-pipeline
                        docker rm ml-pipeline
                    '''
                }
            }
        }

        stage('Create MLflow Server Image') {
            steps {
                script {
                    sh '''
                        # Create Dockerfile for MLflow server
                        cat << EOF > Dockerfile.mlflow
FROM ${DOCKER_IMAGE}:${DOCKER_TAG}

# Copy MLflow data
COPY mlflow_data/ /mlflow/

# Set working directory
WORKDIR /mlflow

# Expose port
EXPOSE 5000

# Start MLflow server
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
EOF

                        # Build and push image
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
                
                To view ML metrics and results:
                docker run -d -p 5001:5000 ${MLFLOW_SERVER_IMAGE}:${DOCKER_TAG}
                Then visit http://localhost:5001
                
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
                # Cleanup
                rm -rf ${WORKSPACE}/mlflow_data
                rm -f Dockerfile.mlflow
                docker logout
            '''
        }
    }
}
