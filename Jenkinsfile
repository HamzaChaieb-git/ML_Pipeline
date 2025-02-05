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
                sh 'python3 -m venv ${VENV_DIR}'
                sh '${PIP} install --upgrade pip'
                sh '${PIP} install -r requirements.txt'
            }
        }

        stage('Linting & Code Quality') {
            steps {
                sh '${PIP} install flake8 black bandit'
                sh 'flake8 . || true'
                sh 'black . || true'
                sh 'bandit -r . || true'
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
        always {
            echo "✅ Pipeline execution complete!"
        }
    }
}
