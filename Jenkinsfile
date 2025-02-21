pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'hamzachaieb01/ml-pipeline'
        DOCKER_TAG = 'latest'
        FINAL_IMAGE = 'hamzachaieb01/ml-trained'
        MLFLOW_IMAGE = 'hamzachaieb01/mlflow-pipeline'
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
                            docker run -d --name linting ${DOCKER_IMAGE}:${DOCKER_TAG} flake8 . || true
                            docker logs -f linting
                            docker rm linting || true
                            
                            docker run -d --name formatting ${DOCKER_IMAGE}:${DOCKER_TAG} black . || true
                            docker logs -f formatting
                            docker rm formatting || true
                            
                            docker run -d --name security ${DOCKER_IMAGE}:${DOCKER_TAG} bandit -r . || true
                            docker logs -f security
                            docker rm security || true
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
                        docker run -d --name mlflow \
                            --network mlflow-net \
                            -p 5000:5000 \
                            -v mlflow_data:/mlflow \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            mlflow ui --host 0.0.0.0 --port 5000
                        sleep 10
                    '''
                }
            }
        }
        
        stage('Prepare Data') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name prepare_data \
                            --network mlflow-net \
                            -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py prepare_data
                        docker logs -f prepare_data || true
                    '''
                }
            }
        }
        
        stage('Train Model') {
            steps {
                timeout(time: 20, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name train_model \
                            --network mlflow-net \
                            -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py train_model
                        docker logs -f train_model || true
                    '''
                }
            }
        }
        
        stage('Evaluate Model') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name evaluate_model \
                            --network mlflow-net \
                            -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py evaluate_model
                        docker logs -f evaluate_model || true
                    '''
                }
            }
        }
        
        stage('Save Model') {
            steps {
                timeout(time: 5, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name save_model \
                            --network mlflow-net \
                            -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py save_model
                        docker logs -f save_model || true
                    '''
                }
            }
        }
        
        stage('Load Model & Re-Evaluate') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name load_model \
                            --network mlflow-net \
                            -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            python main.py load_model
                        docker logs -f load_model || true
                    '''
                }
            }
        }

        stage('Create MLflow Pipeline Image') {
            steps {
                script {
                    writeFile file: 'Dockerfile.mlflow', text: '''
                        FROM ${DOCKER_IMAGE}:${DOCKER_TAG}
                        
                        # Copy MLflow artifacts and data
                        COPY --from=mlflow /mlflow /mlflow
                        COPY --from=load_model /app /app
                        
                        # Set working directory
                        WORKDIR /app
                        
                        # Start script
                        RUN echo '#!/bin/bash\\nmlflow ui --host 0.0.0.0 --port 5000 &\\nsleep 10\\npython main.py all' > /start.sh && \
                            chmod +x /start.sh
                        
                        EXPOSE 5000
                        CMD ["/start.sh"]
                    '''
                    
                    sh '''
                        docker build -t ${MLFLOW_IMAGE}:${DOCKER_TAG} -f Dockerfile.mlflow .
                        docker push ${MLFLOW_IMAGE}:${DOCKER_TAG}
                        echo "✅ MLflow Pipeline image pushed as ${MLFLOW_IMAGE}:${DOCKER_TAG}"
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
                Final ML image: ${FINAL_IMAGE}:${DOCKER_TAG}
                MLflow Pipeline image: ${MLFLOW_IMAGE}:${DOCKER_TAG}
                
                Check console output at $BUILD_URL to view the results.
                
                Changes:
                ${CHANGES}
                
                Failed Tests:
                ${FAILED_TESTS}
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
            echo "Pipeline execution complete!"
            sh '''
                docker rm -f mlflow linting formatting security prepare_data train_model evaluate_model save_model load_model || true
                docker network rm mlflow-net || true
                docker logout || true
                docker system prune -f || true
                rm -f Dockerfile.mlflow || true
            '''
        }
    }
}
