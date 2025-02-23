pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'hamzachaieb01/ml-pipeline'
        DOCKER_TAG = 'latest'
        FINAL_IMAGE = 'hamzachaieb01/ml-trained'
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
        
        stage('Prepare Data') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name prepare_data ${DOCKER_IMAGE}:${DOCKER_TAG} python -m main prepare_data
                        docker logs -f prepare_data || true
                    '''
                }
            }
        }
        
        stage('Train Model') {
            steps {
                timeout(time: 20, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name train_model ${DOCKER_IMAGE}:${DOCKER_TAG} python -m main train_model
                        docker logs -f train_model || true
                    '''
                }
            }
        }
        
        stage('Evaluate Model') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name evaluate_model ${DOCKER_IMAGE}:${DOCKER_TAG} python -m main evaluate_model
                        docker logs -f evaluate_model || true
                    '''
                }
            }
        }
        
        stage('Save Model') {
            steps {
                timeout(time: 5, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name save_model ${DOCKER_IMAGE}:${DOCKER_TAG} python -m main save_model
                        docker logs -f save_model || true
                    '''
                }
            }
        }
        
        stage('Load Model & Re-Evaluate') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name load_model ${DOCKER_IMAGE}:${DOCKER_TAG} python -m main load_model
                        docker logs -f load_model || true
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
                Final image available at: ${FINAL_IMAGE}:${DOCKER_TAG}
                
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
                docker rm -f linting formatting security prepare_data train_model evaluate_model save_model load_model || true
                docker logout || true
                docker system prune -f || true
            '''
        }
    }
}
