    # FROM python:3.13.8
FROM python:3.13.2-bookworm

RUN apt-get update && apt-get install -y \ 
    git \
    curl \
    build-essential \
    jq \
    libsqlite3-dev \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

    
WORKDIR /app

COPY pyproject.toml /app
ADD https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py /app
# RUN pip install uv
# # RUN uv v 
# # RUN . .venv/bin/activate
# RUN pip install --no-cache-dir -r requirements.txt
# RUN uv pip install -r requirements.txt
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN uv lock 

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

ADD . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"


# ADD https://bun.sh/install /bun-install.sh 
# RUN bash /bun-install.sh && rm /bun-install.sh
# ENV PATH="/root/.bun/bin:$PATH"
RUN curl -fsSL https://bun.sh/install | bash && \
    mv /root/.bun/bin/bun /usr/local/bin/bun

# Cache Bun installation by setting environment variables
ENV BUN_INSTALL="/root/.bun"
ENV PATH="$BUN_INSTALL/bin:$PATH"

RUN bun --version

# COPY . /app

EXPOSE 5000

CMD ["uvicorn", "main:app","--host", "0.0.0.0", "--port", "8000"]