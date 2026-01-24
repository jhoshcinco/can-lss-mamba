FROM kirakira2024/mamba2:ssm2.2.6_torch2.5_py3.11

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir pandas scikit-learn

RUN python -c "import pandas, sklearn; print('pandas', pandas.__version__, 'sklearn', sklearn.__version__)"
