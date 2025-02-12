pipeline {
    agent any
    
    environment {
        VENV_DIR = "venv"
        PYTHON = "${VENV_DIR}/bin/python"
        PIP = "${VENV_DIR}/bin/pip"
    }
    
    stages {
        stage('Setup Environment') {
            steps {
                // Clean workspace first
                cleanWs()
                
                // Checkout code
                checkout scm
                
                // Create and activate virtual environment
                sh '''
                    python3 -m venv ${VENV_DIR}
                    ${PIP} install --upgrade pip
                    ${PIP} install -r requirements.txt
                '''
            }
        }
        
        stage('Linting & Code Quality') {
            steps {
                sh '''
                    ${PIP} install flake8 black bandit
                    ${VENV_DIR}/bin/flake8 . || true
                    ${VENV_DIR}/bin/black . || true
                    ${VENV_DIR}/bin/bandit -r . || true
                '''
            }
        }
        
        stage('Prepare Data') {
            steps {
                sh '${PYTHON} main.py prepare_data'
            }
        }
        
        stage('Train Model') {
            steps {
                sh '${PYTHON} main.py train_model'
            }
        }
        
        stage('Evaluate Model') {
            steps {
                sh '${PYTHON} main.py evaluate_model'
            }
        }
        
        stage('Save Model') {
            steps {
                sh '${PYTHON} main.py save_model'
            }
        }
        
        stage('Load Model & Re-Evaluate') {
            steps {
                sh '${PYTHON} main.py load_model'
            }
        }
        
        stage('Run Tests') {
            steps {
                sh '${PYTHON} -m unittest discover -s tests'
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
            // Clean up virtual environment
            sh 'rm -rf ${VENV_DIR}'
            echo "Pipeline execution complete!"
        }
    }
}
