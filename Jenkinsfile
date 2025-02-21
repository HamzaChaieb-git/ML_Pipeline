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
        
        stage('Create MLflow Directory') {
            steps {
                sh '''
                    rm -rf ${WORKSPACE}/mlflow_data
                    mkdir -p ${WORKSPACE}/mlflow_data
                '''
            }
        }
        
        stage('Prepare Data') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name prepare_data \
                            -v ${WORKSPACE}/mlflow_data:/mlflow \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py prepare_data
                        docker logs -f prepare_data || true
                        docker rm prepare_data || true
                    '''
                }
            }
        }
        
        stage('Train Model') {
            steps {
                timeout(time: 20, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name train_model \
                            -v ${WORKSPACE}/mlflow_data:/mlflow \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py train_model
                        docker logs -f train_model || true
                        docker rm train_model || true
                    '''
                }
            }
        }
        
        stage('Evaluate Model') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name evaluate_model \
                            -v ${WORKSPACE}/mlflow_data:/mlflow \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py evaluate_model
                        docker logs -f evaluate_model || true
                        docker rm evaluate_model || true
                    '''
                }
            }
        }
        
        stage('Save Model') {
            steps {
                timeout(time: 5, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name save_model \
                            -v ${WORKSPACE}/mlflow_data:/mlflow \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py save_model
                        docker logs -f save_model || true
                        docker rm save_model || true
                    '''
                }
            }
        }
        
        stage('Load Model & Re-Evaluate') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name load_model \
                            -v ${WORKSPACE}/mlflow_data:/mlflow \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py load_model
                        docker logs -f load_model || true
                        docker rm load_model || true
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
                        echo "✅ MLflow server image created: ${MLFLOW_SERVER_IMAGE}:${DOCKER_TAG}"
                    '''
                }
            }
        }
        
        stage('Save Final Image') {
            steps {
                script {
                    retry(3) {
                        sh '''
                            docker commit load_model ${FINAL_IMAGE}:${DOCKER_TAG}
                            docker push ${FINAL_IMAGE}:${DOCKER_TAG}
                            echo "✅ Final image saved as ${FINAL_IMAGE}:${DOCKER_TAG}"
                        '''
                    }
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
                Final image: ${FINAL_IMAGE}:${DOCKER_TAG}
                MLflow Server: ${MLFLOW_SERVER_IMAGE}:${DOCKER_TAG}
                
                To view metrics & results:
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
                docker rm -f prepare_data train_model evaluate_model save_model load_model || true
                rm -rf ${WORKSPACE}/mlflow_data
                rm -f Dockerfile.mlflow
                docker logout
            '''
        }
    }
}
