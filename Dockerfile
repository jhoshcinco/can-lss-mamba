FROM kirakira2024/mamba2:ssm2.2.6_torch2.5_py3.11

WORKDIR /workspace

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Install dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Verify installation
RUN python -c "import torch, pandas, sklearn, mamba_ssm; print('✅ All core dependencies installed')"

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/results

# Set default command
CMD ["/bin/bash"]
