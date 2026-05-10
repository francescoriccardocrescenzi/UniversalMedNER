# Starting image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="UniboNLP"
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System dependencies (minimal)
RUN apt-get update && apt-get install -y curl git bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create virtual environment and activate it
RUN uv venv /workspace/.venv
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install PyTorch with CUDA support
RUN uv pip install torch torchvision torchaudio \
    --python /workspace/.venv/bin/python \
    --index https://download.pytorch.org/whl/cu122

# Copy only project definition
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install .

CMD ["bash"]