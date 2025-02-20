pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'hamzachaieb01/ml-pipeline'
        DOCKER_TAG = 'latest'
        FINAL_IMAGE = 'hamzachaieb01/ml-trained'
        EMAIL_TO = 'hitthetarget735@gmail.com'
    }
    
    // ... [previous stages remain the same] ...
    
    post {
        success {
            script {
                emailext (
                    subject: "✅ ML Pipeline Success",
                    body: """Pipeline executed successfully!
                    Final image available at: ${FINAL_IMAGE}:${DOCKER_TAG}
                    
                    Check Jenkins for full build logs: ${env.BUILD_URL}""",
                    to: "${EMAIL_TO}",
                    mimeType: 'text/plain',
                    replyTo: "${EMAIL_TO}",
                    attachLog: true,
                    compressLog: true,
                    from: "jenkins@yourdomain.com",  // Replace with your actual domain
                    presendScript: '''
                        println "Sending email to: " + msg.getTo()
                        println "SMTP server: " + msg.getSmtpHost()
                    '''
                )
            }
            echo "✅ Pipeline executed successfully!"
        }
        failure {
            script {
                emailext (
                    subject: "❌ ML Pipeline Failure",
                    body: """Pipeline execution failed!
                    
                    Check Jenkins build logs for details: ${env.BUILD_URL}""",
                    to: "${EMAIL_TO}",
                    mimeType: 'text/plain',
                    replyTo: "${EMAIL_TO}",
                    attachLog: true,
                    compressLog: true,
                    from: "jenkins@yourdomain.com",  // Replace with your actual domain
                    presendScript: '''
                        println "Sending email to: " + msg.getTo()
                        println "SMTP server: " + msg.getSmtpHost()
                    '''
                )
            }
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
