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
                        docker run -d --name prepare_data ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py prepare_data
                        docker logs -f prepare_data || true
                    '''
                }
            }
        }
        
        stage('Train Model') {
            steps {
                timeout(time: 20, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name train_model ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py train_model
                        docker logs -f train_model || true
                    '''
                }
            }
        }
        
        stage('Evaluate Model') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name evaluate_model ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py evaluate_model
                        docker logs -f evaluate_model || true
                    '''
                }
            }
        }
        
        stage('Save Model') {
            steps {
                timeout(time: 5, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name save_model ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py save_model
                        docker logs -f save_model || true
                    '''
                }
            }
        }
        
        stage('Load Model & Re-Evaluate') {
            steps {
                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        docker run -d --name load_model ${DOCKER_IMAGE}:${DOCKER_TAG} python main.py load_model
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
        // ... [previous stages remain the same] ...
    }
    
    post {
        success {
            script {
                def finalImage = "${FINAL_IMAGE}:${DOCKER_TAG}"
                emailext (
                    subject: "ML Pipeline - Build #${BUILD_NUMBER} - Success",
                    body: """
                        <html>
                        <body>
                            <div style="font-family: Arial, sans-serif; padding: 20px;">
                                <h2 style="color: #2ecc71;">✅ BUILD SUCCESS</h2>
                                <hr/>
                                
                                <h3>Build Information</h3>
                                <ul>
                                    <li><b>Project:</b> ${JOB_NAME}</li>
                                    <li><b>Build Number:</b> #${BUILD_NUMBER}</li>
                                    <li><b>Build URL:</b> <a href="${BUILD_URL}">${BUILD_URL}</a></li>
                                    <li><b>Duration:</b> ${currentBuild.durationString}</li>
                                </ul>
                                
                                <h3>Pipeline Results</h3>
                                <p>Pipeline executed successfully!</p>
                                <p><b>Final image:</b> ${finalImage}</p>
                                
                                <hr/>
                                <p style="font-size: 12px; color: #666;">
                                    This is an automated email from Jenkins CI/CD Pipeline.<br/>
                                    Please do not reply to this email.
                                </p>
                            </div>
                        </body>
                        </html>
                    """,
                    to: "${EMAIL_TO}",
                    mimeType: 'text/html'
                )
            }
            echo "✅ Pipeline executed successfully!"
            echo "Final image available at: ${FINAL_IMAGE}:${DOCKER_TAG}"
        }
        failure {
            script {
                def finalImage = "${FINAL_IMAGE}:${DOCKER_TAG}"
                emailext (
                    subject: "ML Pipeline - Build #${BUILD_NUMBER} - Failed",
                    body: """
                        <html>
                        <body>
                            <div style="font-family: Arial, sans-serif; padding: 20px;">
                                <h2 style="color: #e74c3c;">❌ BUILD FAILED</h2>
                                <hr/>
                                
                                <h3>Build Information</h3>
                                <ul>
                                    <li><b>Project:</b> ${JOB_NAME}</li>
                                    <li><b>Build Number:</b> #${BUILD_NUMBER}</li>
                                    <li><b>Build URL:</b> <a href="${BUILD_URL}">${BUILD_URL}</a></li>
                                    <li><b>Duration:</b> ${currentBuild.durationString}</li>
                                </ul>
                                
                                <h3>Failure Information</h3>
                                <p>Pipeline execution failed!</p>
                                <p>Please check the build logs at the URL above for more details.</p>
                                
                                <hr/>
                                <p style="font-size: 12px; color: #666;">
                                    This is an automated email from Jenkins CI/CD Pipeline.<br/>
                                    Please do not reply to this email.
                                </p>
                            </div>
                        </body>
                        </html>
                    """,
                    to: "${EMAIL_TO}",
                    mimeType: 'text/html'
                )
            }
            echo "❌ Pipeline failed!"
            echo "Check the logs above for details"
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
