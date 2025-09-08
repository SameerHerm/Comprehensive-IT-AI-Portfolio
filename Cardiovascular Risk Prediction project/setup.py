"""
Setup configuration for Cardiovascular Risk Prediction package
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Separate core requirements from optional ones
core_requirements = []
dev_requirements = []
ml_requirements = []

for req in requirements:
    if req.startswith('#') or req.strip() == '':
        continue
    elif any(pkg in req for pkg in ['pytest', 'black', 'flake8', 'pylint', 'mypy']):
        dev_requirements.append(req)
    elif any(pkg in req for pkg in ['tensorflow', 'torch', 'keras']):
        ml_requirements.append(req)
    else:
        core_requirements.append(req)

setup(
    name='cardiovascular-risk-prediction',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A comprehensive machine learning system for cardiovascular risk prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SameerHerm/Comprehensive-IT-AI-Portfolio',
    project_urls={
        'Bug Reports': 'https://github.com/SameerHerm/Comprehensive-IT-AI-Portfolio/issues',
        'Source': 'https://github.com/SameerHerm/Comprehensive-IT-AI-Portfolio',
        'Documentation': 'https://github.com/SameerHerm/Comprehensive-IT-AI-Portfolio/tree/main/Cardiovascular%20Risk%20Prediction%20project',
    },
    
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    python_requires='>=3.8',
    
    install_requires=core_requirements,
    
    extras_require={
        'dev': dev_requirements,
        'ml': ml_requirements,
        'all': dev_requirements + ml_requirements,
    },
    
    entry_points={
        'console_scripts': [
            'cvr-train=src.train:main',
            'cvr-predict=src.predict:main',
            'cvr-evaluate=src.evaluate:main',
            'cvr-webapp=web_app.app:main',
        ],
    },
    
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.json', '*.csv'],
        'web_app': ['templates/*.html', 'static/*'],
        'models': ['*.pkl', '*.h5', '*.joblib'],
    },
    
    data_files=[
        ('config', ['config/config.yaml', 'config/logging.yaml']),
        ('data/raw', []),
        ('data/processed', []),
        ('models', []),
        ('logs', []),
    ],
    
    zip_safe=False,
    
    test_suite='tests',
    tests_require=[
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'pytest-mock>=3.11.1',
    ],
    
    keywords=[
        'machine learning',
        'cardiovascular risk',
        'health prediction',
        'medical ai',
        'risk assessment',
        'predictive modeling',
        'healthcare analytics',
    ],
)

# Post-installation script
def post_install():
    """Create necessary directories after installation"""
    import os
    
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'models',
        'logs',
        'reports',
        'notebooks',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create .gitkeep files to preserve directory structure
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            open(gitkeep_path, 'a').close()

if __name__ == '__main__':
    import sys
    if 'install' in sys.argv:
        import atexit
        atexit.register(post_install)
