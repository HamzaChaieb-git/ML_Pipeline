stage('Pull Docker Image') {
    steps {
        timeout(time: 10, unit: 'MINUTES') {
            retry(3) {
                script {
                    echo "Pulling image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
                    sh '''
                        docker login -u hamzachaieb01 --password-stdin < /dev/null || true
                        docker pull ${DOCKER_IMAGE}:${DOCKER_TAG} || true
                    '''
                }
            }
        }
    }
}
